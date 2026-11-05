"""Archives the Deno runner starts Pyodide from, built once on the host under ``<dspy cache dir>/sandbox``.

Pyodide otherwise compiles every module it imports, its own standard library included, on every start. A Deno run
that executes no sandboxed code byte-compiles each archive once, so sandboxes start in a fraction of the time:

- ``stdlib.zip``: Pyodide's standard library, which every ``PythonInterpreter`` starts from. It depends only on the
  Pyodide release.
- ``dspy-<digest>.zip``, for sandboxes whose code imports dspy (``PythonInterpreter.provide_dspy``), which have no
  network to install packages with. It holds dspy and every package its installed metadata says it needs: those
  Pyodide ships as Pyodide builds them (checked against Pyodide's lock file), the others from this environment if
  they are pure Python. There is one per set of sources.

``DSPY_SANDBOX_CACHE=0`` or ``PythonInterpreter(sandbox_cache=False)`` turns the cache off, for environments where
writing files is unwelcome: nothing is written under the cache directory. Pyodide then starts from its own standard
library, and a sandbox that imports dspy, which needs the bundle, gets one built in a temporary directory of the
process, removed when it exits.
"""

from __future__ import annotations

import atexit
import contextlib
import fnmatch
import functools
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from dspy.primitives.deno_utils import deno_subprocess_env
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
_CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
# Bump when the bundle layout changes.
_FORMAT = 3
# dspy's package data, as pyproject.toml's [tool.setuptools.package-data] declares it for builds, relative to the
# package. A checkout outside a git work tree still bundles it, with its Python source.
_DSPY_PACKAGE_DATA = (
    "primitives/*.js",
    "clients/model_metadata/snapshot.json.gz",
    "clients/model_metadata/LICENSE",
    "clients/model_metadata/provenance.json",
    "_vendor/lm15-LICENSE",
    "_vendor/lm15-provenance.txt",
    "_vendor/UPDATING.md",
    "_vendor/lm15/py.typed",
)
_COMPILED = {".so", ".pyd", ".dylib"}
_lock = threading.Lock()
_failed: dict[tuple, tuple[str, float]] = {}
_RETRY_AFTER = 60.0


class SandboxBundleError(DSPyError):
    """A sandbox archive could not be prepared (for example, the dspy bundle offline before its first download)."""


def sandbox_cache_env_enabled() -> bool:
    """Whether interpreters may write the sandbox cache, unless ``DSPY_SANDBOX_CACHE`` turns it off (``0``,
    ``false``, ``no`` or ``off``)."""
    return os.environ.get("DSPY_SANDBOX_CACHE", "").strip().lower() not in ("0", "false", "no", "off")


@functools.cache
def scratch_root() -> Path:
    """A temporary directory of this process for the dspy bundle when the cache is off, removed when it exits."""
    root = Path(tempfile.mkdtemp(prefix="dspy-sandbox-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def bundle_root() -> Path:
    return Path(DSPY_CACHEDIR) / "sandbox" / f"pyodide-{PYODIDE_VERSION}"


def prepare_stdlib(deno_command: list[str], deno_dir: str | None, runner: str) -> Path:
    """Return the byte-compiled standard library, building it if it is missing. Thread-safe."""
    with _lock:
        return _guarded(_prepare_stdlib, tuple(deno_command), deno_dir, runner)


def prepare_bundle(deno_command: list[str], deno_dir: str | None, runner: str, root: Path | None = None) -> Path:
    """Return the dspy bundle for the current sources, building it in ``root`` (by default the cache) if it is
    missing. Thread-safe."""
    with _lock:
        return _guarded(_prepare_bundle, tuple(deno_command), deno_dir, runner, root or bundle_root())


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


def _prepare_stdlib(deno_command: tuple[str, ...], deno_dir: str | None, runner: str) -> Path:
    target = bundle_root() / "stdlib.zip"
    if not target.exists():
        _build(deno_command, deno_dir, runner, target)
    return target


def _prepare_bundle(deno_command: tuple[str, ...], deno_dir: str | None, runner: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    wheels, files = _contents(root, deno_dir)
    stamp = [_FORMAT, PYODIDE_VERSION, wheels, [(name, path.stat().st_size, path.stat().st_mtime_ns) for path, name in files]]
    target = root / f"dspy-{hashlib.sha256(json.dumps(stamp).encode()).hexdigest()[:16]}.zip"
    if target.exists():
        os.utime(target)  # in use: keep it from being pruned
        return target
    source = root / f"source.{os.getpid()}.{threading.get_ident()}.zip"
    try:
        with zipfile.ZipFile(source, "w") as archive:
            for wheel in wheels:  # a wheel is installed by unpacking it
                with zipfile.ZipFile(root / wheel) as contents:
                    for item in contents.infolist():
                        archive.writestr(item, contents.read(item))
            for path, name in files:
                archive.write(path, name)
        _build(deno_command, deno_dir, runner, target, source)
    finally:
        source.unlink(missing_ok=True)
    _prune(root, keep=target)
    return target


def _build(deno_command: tuple[str, ...], deno_dir: str | None, runner: str, target: Path, source: Path | None = None):
    """Have the runner byte-compile ``source`` (by default Pyodide's standard library) into ``target``. The run
    executes no sandboxed code and may write only a temporary file, which then replaces ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    args = [
        *deno_command,
        f"--allow-read={','.join(str(p) for p in (deno_dir, runner, source) if p)}",
        f"--allow-write={temp}",
        runner,
        "",
        *([f"--dspy-deno-dir={deno_dir}"] if deno_dir else []),
        f"--dspy-build={temp}",
        *([f"--dspy-build-from={source}"] if source else []),
    ]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600, env=deno_subprocess_env())
        if result.returncode != 0:
            raise SandboxBundleError(f"Building a Pyodide sandbox archive failed:\n{result.stdout}\n{result.stderr}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _contents(root: Path, deno_dir: str | None) -> tuple[list[str], list[tuple[Path, str]]]:
    """The Pyodide wheels and the files of this environment that make up the bundle."""
    lock, needed, files = _plan(deno_dir)
    if _planned_at.setdefault(deno_dir, _folder_times(files)) != _folder_times(files):
        # A file was added, removed or renamed in a folder the plan lists files from: list them again.
        _plan.cache_clear()
        lock, needed, files = _plan(deno_dir)
        _planned_at[deno_dir] = _folder_times(files)
    return [_fetch_wheel(root, lock[name], deno_dir) for name in needed], files


# The modification times of the folders each plan's files are in, when it was made.
_planned_at: dict[str | None, list[int | None]] = {}


def _folder_times(files: list[tuple[Path, str]]) -> list[int | None]:
    """The modification times of the folders the files are in, from each file's up to its package's own."""
    folders = {path.parents[level] for path, name in files for level in range(max(1, name.count("/")))}
    times = []
    for folder in sorted(folders):
        try:
            times.append(folder.stat().st_mtime_ns)
        except OSError:  # removed
            times.append(None)
    return times


@functools.cache
def _plan(deno_dir: str | None) -> tuple[dict, list[str], list[tuple[Path, str]]]:
    """Which packages come from Pyodide and which files from this environment, once per process and again when the
    files' folders change (see ``_contents``). The files' sizes and times are read on every call, so an edited file
    still makes a new bundle."""
    lock = _pyodide_lock(deno_dir)
    shipped, files = [], []
    for dist in _distributions():
        name = _normalize(dist.metadata["Name"])
        if name in lock:
            shipped.append(name)
        else:
            files += _files(dist)
    needed: list[str] = []
    while shipped:  # with what they need, by Pyodide's lock file
        name = _normalize(shipped.pop())
        if name not in needed:
            needed.append(name)
            shipped += lock[name]["depends"]
    return lock, sorted(needed), files


def _distributions() -> list[metadata.Distribution]:
    """dspy's distribution and every one it requires, at any depth, from this environment's metadata."""
    from dspy.utils.lazy_import import _detect_dspy_dist

    found: dict[str, metadata.Distribution] = {}
    todo = [_detect_dspy_dist()]
    while todo:
        name = _normalize(todo.pop())
        if name in found:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:  # one this platform does not need
            continue
        found[name] = dist
        todo += [re.match(r"[\w.-]+", requirement).group(0) for requirement in dist.requires or ()
                 if "extra ==" not in requirement]
    return list(found.values())


def _top_level(dist: metadata.Distribution) -> set[str]:
    """A distribution's top-level modules, from its own metadata."""
    listed = (dist.read_text("top_level.txt") or "").split() or [
        file.parts[0].removesuffix(".py") for file in dist.files or ()
        if file.suffix == ".py" and not file.parts[0].endswith((".dist-info", ".data"))
    ]
    tops = {top for top in listed if not top.startswith("__")}  # not an editable install's finder
    return tops or {_normalize(dist.metadata["Name"]).replace("-", "_")}  # an editable install lists no modules


def _files(dist: metadata.Distribution) -> list[tuple[Path, str]]:
    """A pure-Python distribution's modules, from where they are imported here (the host's own dspy, be it a checkout
    or an install). An install's RECORD lists them when they are imported from it, otherwise they are found on disk. A
    compiled distribution is left out: Pyodide cannot run it."""
    out = []
    for top in sorted(_top_level(dist)):
        spec = importlib.util.find_spec(top)
        if spec is None:
            continue
        if spec.submodule_search_locations:
            where, name = Path(next(iter(spec.submodule_search_locations))).resolve(), top
        elif spec.origin and spec.origin.endswith(".py"):
            where, name = Path(spec.origin).resolve(), f"{top}.py"
        else:
            return []
        # A wheel install's RECORD lists each file (an egg-info's listing can be out of date).
        recorded = [file for file in dist.files or () if file.parts[0] == name] if dist.read_text("RECORD") else []
        if recorded and Path(dist.locate_file(name)).resolve() == where:
            paths = [(Path(dist.locate_file(file)), file.as_posix()) for file in recorded]
        elif where.is_file():
            paths = [(where, name)]
        else:
            data = _DSPY_PACKAGE_DATA if top == "dspy" else ()
            paths = [(path, f"{top}/{path.relative_to(where).as_posix()}") for path in _shipped(where, data)]
        if any(path.suffix in _COMPILED for path, _ in paths):
            return []
        out += [(path, arcname) for path, arcname in paths
                if "__pycache__" not in path.parts and path.suffix != ".pyc"]
    return out


def _shipped(where: Path, data: tuple[str, ...] = ()) -> list[Path]:
    """The files of a package directory on disk to bundle: its Python source, unless git ignores it, and the other
    files git tracks, such as package data. Hidden files are left out. Outside a git work tree, so is anything but
    Python source and the declared package ``data`` (globs relative to ``where``), since local files like credentials
    may sit next to it. An installed package lists its own."""
    tracked, untracked = _git_files(where), _git_files(where, "--others", "--exclude-standard")
    shipped = []
    for path in sorted(where.rglob("*")):
        relative = path.relative_to(where)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if tracked is None or untracked is None:
            keep = path.suffix == ".py" or any(fnmatch.fnmatchcase(relative.as_posix(), pattern) for pattern in data)
        else:
            keep = relative.as_posix() in tracked or (path.suffix == ".py" and relative.as_posix() in untracked)
        if keep:
            shipped.append(path)
    return shipped


def _git_files(where: Path, *options: str) -> set[str] | None:
    """The files below ``where`` that ``git ls-files`` lists with ``options``, or ``None`` outside a git work tree."""
    try:
        listed = subprocess.run(
            ["git", "-C", str(where), "ls-files", "-z", *options], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):  # no git
        return None
    return set(listed.stdout.split("\0")) - {""} if listed.returncode == 0 else None


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _prune(root: Path, keep: Path, count: int = 3, min_age: float = 3600) -> None:
    """Remove bundles beyond the ``count`` most recently used that no process has used for ``min_age`` seconds, e.g.
    those built from earlier dspy sources. Other processes build and replace files here at the same time: their
    temporary files are left alone, a file that is gone by the time it is looked at is skipped, and pruning never
    fails the build it follows."""
    bundles = sorted(
        ((used, path) for path in root.glob("dspy-*") if path.suffix != ".tmp" and (used := _last_used(path))),
        reverse=True,
    )
    for _, old in [(used, path) for used, path in bundles if path != keep][count - 1 :]:
        # Read again right before: another process may have just taken it into use.
        if (used := _last_used(old)) and time.time() - used > min_age:
            with contextlib.suppress(OSError):
                shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink(missing_ok=True)


def _last_used(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:  # removed or replaced by another process meanwhile
        return None


def _pyodide_lock(deno_dir: str | None) -> dict:
    local = _npm_dir(deno_dir) / "pyodide-lock.json" if deno_dir else None
    if local is not None and local.exists():
        data = local.read_text()
    else:
        data = _download(_CDN + "pyodide-lock.json").decode()
    return {_normalize(name): info for name, info in json.loads(data)["packages"].items()}


def _npm_dir(deno_dir: str) -> Path:
    return Path(deno_dir) / "npm" / "registry.npmjs.org" / "pyodide" / PYODIDE_VERSION


def _fetch_wheel(root: Path, info: dict, deno_dir: str | None) -> str:
    name = info["file_name"]
    target = root / name
    if not target.exists():
        cached = _npm_dir(deno_dir) / name if deno_dir else None
        data = cached.read_bytes() if cached is not None and cached.exists() else _download(_CDN + name)
        if hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise SandboxBundleError(f"{name} does not match the hash in Pyodide's lock file")
        temp = target.with_name(f"{name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp.write_bytes(data)
        os.replace(temp, target)
    return name


def _download(url: str) -> bytes:
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content
