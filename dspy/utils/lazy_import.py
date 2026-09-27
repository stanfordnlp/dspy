"""Lazy-import helpers for optional dependencies.

Optional deps must be importable lazily so that `import dspy` succeeds even
when they are absent. Call sites get a module-level binding that defers the
real import until first attribute access:

    from dspy.utils.lazy_import import require

    np = require("numpy")          # zero cost -- no import happens here
    np.array([1, 2, 3])            # numpy is loaded on first use

If the package is not installed, the first attribute access raises
`ImportError` with an install hint.

Lazy modules are materialized under a per-module lock so concurrent first use
cannot expose a partially initialized module.
"""

import functools
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import sys
import threading
import types
from typing import Any


def _detect_dspy_dist() -> str:
    for dist in ("dspy", "dspy-ai"):
        try:
            importlib.metadata.version(dist)
            return dist
        except importlib.metadata.PackageNotFoundError:
            continue
    return "dspy"

_INSTALL_HINTS: dict[str, str] = {
    "optuna": "optuna",
    "mcp": "mcp",
    "langchain_core": "langchain",
    "weaviate": "weaviate",
    "anthropic": "anthropic",
    "numpy": "numpy",
    "litellm": "litellm",
}


_lazy_module_locks: dict[str, threading.RLock] = {}
_lazy_module_locks_lock = threading.Lock()


def _get_lazy_module_lock(module: str) -> threading.RLock:
    with _lazy_module_locks_lock:
        return _lazy_module_locks.setdefault(module, threading.RLock())


class _MissingModule(types.ModuleType):
    """Stand-in returned by `require()` when a package is not installed.

    Raises `ImportError` with an install hint on any attribute access.
    Records the original call site so the traceback is actionable.
    """

    def __init__(self, module: str, message: str, frame_data: dict):
        super().__init__(module)
        self._message = message
        self._frame_data = frame_data

    def __getattr__(self, attr: str):
        fd = self._frame_data
        raise ImportError(
            f"{self._message}\n\n"
            "This error is lazily reported, having originally occurred in\n"
            f"  File {fd['filename']}, line {fd['lineno']}, in {fd['function']}\n\n"
            f"----> {''.join(fd['code_context'] or '').strip()}"
        )


class _LazyModule(types.ModuleType):
    """Module proxy that imports the real module on first attribute access.

    Attribute assignment also materializes the real module so configuration writes apply to the real dependency.
    Assigning a *module* value is treated as the import system binding a submodule onto
    this (parent) proxy and never triggers a load; the binding is replayed onto the real
    module once it materializes.
    """

    def __init__(self, module: str, spec: importlib.machinery.ModuleSpec, lock: threading.RLock):
        super().__init__(module)
        self.__spec__ = spec
        self.__loader__ = spec.loader
        self.__package__ = spec.parent
        if spec.submodule_search_locations is not None:
            self.__path__ = spec.submodule_search_locations
        self._dspy_lazy_spec = spec
        self._dspy_lazy_lock = lock

    def _load(self) -> types.ModuleType:
        # The proxy starts in sys.modules, then the first attribute access swaps in and executes the real module under
        # the per-module lock. If import fails, restore the proxy so later accesses can retry and still share the lock.
        # Return sys.modules after execution because a module may replace itself while importing.
        module_name = self.__name__
        with self._dspy_lazy_lock:
            loaded = sys.modules.get(module_name)
            if loaded is not None and loaded is not self:
                return loaded

            spec = self._dspy_lazy_spec
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules[module_name] = self
                raise
            real = sys.modules.get(module_name, module)
            self._replay_deferred_submodules(real)
            return real

    def _replay_deferred_submodules(self, real: types.ModuleType) -> None:
        # Submodules the import system bound onto the proxy before the real module
        # materialized (see __setattr__). The real module's own imports found those
        # submodules already in sys.modules, so the import system never re-bound them
        # onto it; copy the bindings over so `real.<submodule>` resolves the same way
        # it would have without the proxy in the middle.
        for attr, value in list(self.__dict__.items()):
            if isinstance(value, types.ModuleType) and not hasattr(real, attr):
                setattr(real, attr, value)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_dspy_lazy_") or attr in {"__spec__", "__loader__", "__package__", "__path__"}:
            super().__setattr__(attr, value)
        elif isinstance(value, types.ModuleType):
            # The import system binds submodules onto their parent package with
            # setattr(). Materializing the real module from here would re-enter the
            # import machinery while this package's submodule can still be
            # mid-initialization (e.g. `import numpy._core` while sys.modules still
            # holds this proxy), nesting a full exec of the parent inside the
            # submodule's init and corrupting it. Bind locally instead; _load()
            # replays these bindings onto the real module once it materializes.
            super().__setattr__(attr, value)
            # A caller can retain the proxy past first use. Once the real module has
            # materialized, sys.modules no longer routes through this object, so the
            # import system binds submodules onto the real module directly — any
            # module-valued assignment reaching the retained proxy from here on can
            # only be caller code and must replace the canonical module's binding,
            # exactly like a non-module configuration write.
            loaded = sys.modules.get(self.__name__)
            if loaded is not None and loaded is not self:
                setattr(loaded, attr, value)
        else:
            setattr(self._load(), attr, value)

    def __dir__(self) -> list[str]:
        return dir(self._load())


@functools.cache
def is_available(module: str) -> bool:
    """Return True if *module* can be imported, without actually importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, *, extra: str | None = None, feature: str | None = None) -> Any:
    """Return a lazily-loaded module, or a stub that raises on access.

    Safe to call at module level:

        np = require("numpy")

    **Installed** -- returns a `_LazyModule` proxy. The real import happens on first attribute access, guarded by a
    per-module lock so concurrent first use cannot observe a partially initialized module.

    **Not installed** -- returns a `_MissingModule` stub. The first attribute
    access raises `ImportError` with a `pip install dspy[…]` hint and the
    file/line where `require()` was originally called.

    Args:
        module: Dotted module path (e.g. `"numpy"`).
        extra: Name of the dspy extra that provides this dep.
        feature: Label shown in the error (e.g. `"dspy.Embeddings"`).
    """
    lock = _get_lazy_module_lock(module)
    with lock:
        if module in sys.modules:
            return sys.modules[module]

        spec = importlib.util.find_spec(module)
    if spec is None or spec.loader is None:
        top = module.split(".", 1)[0]
        feat = feature or "this feature"
        ext = extra or _INSTALL_HINTS.get(top, top)
        dist = _detect_dspy_dist()
        message = (
            f"{top} is required to use {feat}. "
            f"Install with `pip install {dist}[{ext}]` or `pip install {top}`."
        )
        parent = inspect.stack()[1]
        frame_data = {
            "filename": parent.filename,
            "lineno": parent.lineno,
            "function": parent.function,
            "code_context": parent.code_context,
        }
        del parent
        return _MissingModule(module, message, frame_data)

    with lock:
        if module in sys.modules:
            return sys.modules[module]

        mod = _LazyModule(module, spec, lock)
        sys.modules[module] = mod
        return mod
