"""Pyodide's standard library with its bytecode, which the Deno runner starts Pyodide from.

Pyodide otherwise compiles every module of its standard library it imports, on every start. The host builds
``<dspy cache dir>/sandbox/pyodide-<release>/stdlib.zip`` once per Pyodide release, in one Deno run that executes no
sandboxed code, and every ``PythonInterpreter`` starts from it, in a fraction of the time. ``DSPY_SANDBOX_CACHE=0``
or ``PythonInterpreter(sandbox_cache=False)`` turns this off, for environments where writing files is unwelcome.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

from dspy.utils.caching import DSPY_CACHEDIR
from dspy.utils.exceptions import DSPyError


def _pinned_pyodide(runner: Path = Path(__file__).parent / "runner.js") -> str | None:
    """The Pyodide release runner.js imports, where it is pinned: upgrading Pyodide there also moves the cache.
    ``None`` where the runner can't be read, which then can't start an interpreter either, so that ``import dspy``
    still works there."""
    try:
        return re.search(r"npm:pyodide@([\w.-]+)/", runner.read_text()).group(1)
    except (OSError, AttributeError):
        return None


PYODIDE_VERSION = _pinned_pyodide()
_lock = threading.Lock()
_failed: dict[tuple, tuple[str, float]] = {}
_RETRY_AFTER = 60.0


class SandboxBundleError(DSPyError):
    """A sandbox archive could not be prepared."""


def cache_enabled() -> bool:
    """Whether interpreters may write the sandbox cache, unless ``DSPY_SANDBOX_CACHE`` turns it off (``0``,
    ``false``, ``no`` or ``off``)."""
    return os.environ.get("DSPY_SANDBOX_CACHE", "").strip().lower() not in ("0", "false", "no", "off")


def bundle_root() -> Path:
    return Path(DSPY_CACHEDIR) / "sandbox" / f"pyodide-{PYODIDE_VERSION}"


def prepare_stdlib(deno_command: list[str], deno_dir: str | None, runner: str) -> Path:
    """Return the byte-compiled standard library, building it if it is missing. Thread-safe."""
    with _lock:
        return _guarded(_prepare_stdlib, tuple(deno_command), deno_dir, runner)


def _guarded(prepare, *args):
    # The archives are checked on every call, so a cleared cache gets rebuilt. A failed build is retried only after
    # _RETRY_AFTER seconds: callers in between fail fast (or fall back) instead of each paying for another attempt,
    # and a long-running process recovers once the cause, like a network outage, goes away.
    key = (prepare.__name__, *args)
    failed = _failed.get(key)
    if failed is not None and time.monotonic() - failed[1] < _RETRY_AFTER:
        raise SandboxBundleError(failed[0])
    try:
        result = prepare(*args)
    except Exception as e:
        message = str(e) if isinstance(e, SandboxBundleError) else f"Preparing the Pyodide sandbox failed: {e}"
        _failed[key] = (message, time.monotonic())
        if isinstance(e, SandboxBundleError):
            raise
        raise SandboxBundleError(message) from e
    _failed.pop(key, None)
    return result


def _build(deno_command: tuple[str, ...], deno_dir: str | None, runner: str, read: list[str], write: Path,
           flags: list[str]) -> None:
    """Run the runner in a build mode, which runs no sandboxed code and may write only ``write``."""
    from dspy.primitives.python_interpreter import _deno_subprocess_env

    args = [
        *deno_command,
        f"--allow-read={','.join(p for p in (deno_dir, runner, *read) if p)}",
        f"--allow-write={write}",
        runner,
        "",
        *([f"--dspy-deno-dir={deno_dir}"] if deno_dir else []),
        *flags,
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=600, env=_deno_subprocess_env())
    if result.returncode != 0:
        raise SandboxBundleError(f"Building a Pyodide sandbox archive failed:\n{result.stdout}\n{result.stderr}")


def _prepare_stdlib(deno_command: tuple[str, ...], deno_dir: str | None, runner: str) -> Path:
    root = bundle_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "stdlib.zip"
    if not target.exists():
        temp = root / f"stdlib.{os.getpid()}.{threading.get_ident()}.zip"
        try:
            _build(deno_command, deno_dir, runner, [], temp, [f"--dspy-build-stdlib={temp}"])
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    return target
