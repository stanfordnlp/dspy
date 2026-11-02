"""Pyodide's standard library with its bytecode, which the Deno runner starts Pyodide from.

Pyodide otherwise compiles every module of its standard library it imports, on every start. The host builds
``<dspy cache dir>/sandbox/pyodide-<release>/stdlib.zip`` once per Pyodide release, in one Deno run that executes no
sandboxed code, and every ``PythonInterpreter`` starts from it, in a fraction of the time.
"""

from __future__ import annotations

import functools
import os
import subprocess
import threading
from pathlib import Path

from dspy.utils.caching import DSPY_CACHEDIR
from dspy.utils.exceptions import DSPyError

# Must match the npm:pyodide version runner.js imports.
PYODIDE_VERSION = "0.29.4"
_lock = threading.Lock()


class SandboxBundleError(DSPyError):
    """A sandbox archive could not be prepared."""


def bundle_root() -> Path:
    return Path(DSPY_CACHEDIR) / "sandbox" / f"pyodide-{PYODIDE_VERSION}"


def prepare_stdlib(deno_command: list[str], deno_dir: str | None, runner: str) -> Path:
    """Return the byte-compiled standard library, building it if missing. Thread-safe and cached per process."""
    with _lock:
        return _guarded(_prepare_stdlib, tuple(deno_command), deno_dir, runner)


def _guarded(prepare, *args):
    try:
        return prepare(*args)
    except SandboxBundleError:
        raise
    except Exception as e:
        raise SandboxBundleError(f"Preparing the Pyodide sandbox failed: {e}") from e


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


@functools.cache
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
