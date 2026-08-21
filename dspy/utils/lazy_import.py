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


class _ReuseExecutedLoader:
    """Loader that hands back an already-executed module unchanged.

    The import machinery does not re-check ``sys.modules`` once a direct
    child import has committed to a spec, so a child that completed during
    parent materialization (the parent ``__init__`` imports it) would be
    re-executed by the original pending import. This loader makes that
    pending import resolve to the completed module object instead, and
    restores its original ``__spec__``/``__loader__`` afterwards.
    """

    def __init__(self, module: types.ModuleType):
        self._module = module
        self._original_spec = module.__spec__
        self._original_loader = module.__loader__
        self._original_path: list[str] | None = list(getattr(module, "__path__", [])) or None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        return self._module

    def exec_module(self, module: types.ModuleType) -> None:
        module.__spec__ = self._original_spec
        module.__loader__ = self._original_loader
        # spec_from_loader(is_package=True) pins an empty search path, which
        # module_from_spec would have stamped onto the reused package --
        # restore the real one so later submodule imports still resolve.
        if self._original_path is not None:
            module.__path__ = self._original_path


class _MaterializedChildFinder:
    """Meta-path finder deduplicating children imported during materialization.

    Returns a reuse spec only for a module that is already in
    ``sys.modules``, was imported while a lazy parent was materializing,
    and is therefore the subject of an outer pending import; every other
    lookup falls through to the normal finders.
    """

    _MARKER = "_dspy_lazy_reuse"

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001, ANN202
        # A reload passes the module as ``target`` and must re-execute it;
        # reuse only serves the one pending import that raced the parent's
        # materialization.
        if target is not None:
            return None
        module = sys.modules.get(fullname)
        if module is None or getattr(module, _MaterializedChildFinder._MARKER, False) is not True:
            return None
        # Consume the marker: the reuse serves exactly the pending import,
        # and a later reload or re-import must go through the normal path.
        try:
            delattr(module, _MaterializedChildFinder._MARKER)
        except AttributeError:
            pass
        return importlib.util.spec_from_loader(
            fullname,
            _ReuseExecutedLoader(module),
            is_package=hasattr(module, "__path__"),
        )


sys.meta_path.insert(0, _MaterializedChildFinder())

# True while a lazy parent's real module executes: modules imported in that
# window may be the subject of the outer pending import that triggered
# materialization, so they are marked for reuse by _load().
_MATERIALIZING = False


def _is_submodule_binding(package: str, attr: str, value: Any) -> bool:
    """Return True when *value* is the import machinery binding a submodule.

    The machinery binds ``sys.modules[parent].<child> = <module>`` after a
    submodule import; such a value is a module whose ``__name__`` is the
    child's fully qualified name. Any other assignment (configuration writes
    from dspy call sites) keeps the materialize-on-assign behaviour.
    """
    return isinstance(value, types.ModuleType) and getattr(value, "__name__", None) == f"{package}.{attr}"


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
    """

    def __init__(self, module: str, spec: importlib.machinery.ModuleSpec, lock: threading.RLock):
        super().__init__(module)
        self.__spec__ = spec
        self.__loader__ = spec.loader
        self.__package__ = spec.parent
        self._dspy_lazy_spec = spec
        self._dspy_lazy_lock = lock

    @property
    def __path__(self) -> Any:
        # The import machinery reads the parent package's __path__ to locate
        # and import submodules, before any submodule code runs. Answering
        # from the proxy would let a direct submodule import (a C extension
        # doing `import numpy._core` after `import dspy`) execute the child
        # while the real package __init__ never ran -- its side effects
        # (numpy registers its Windows DLL directory, then re-exports _core)
        # are skipped and the child crashes. Materializing here runs the
        # real package first, exactly where the machinery would have.
        return getattr(self._load(), "__path__")

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
            global _MATERIALIZING
            prior_materializing = _MATERIALIZING
            imported_before = set(sys.modules)
            _MATERIALIZING = True
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules[module_name] = self
                raise
            finally:
                _MATERIALIZING = prior_materializing
            # Modules the package __init__ imported may be the subject of the
            # outer pending import that triggered this materialization; mark
            # them so the machinery's committed import reuses the completed
            # module instead of re-executing it.
            for imported_name, imported_module in sys.modules.items():
                if imported_name not in imported_before:
                    try:
                        setattr(imported_module, _MaterializedChildFinder._MARKER, True)
                    except (AttributeError, TypeError):
                        pass
            return sys.modules.get(module_name, module)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_dspy_lazy_") or attr in {"__spec__", "__loader__", "__package__"}:
            super().__setattr__(attr, value)
        elif _is_submodule_binding(self.__name__, attr, value):
            # The import machinery binds a freshly imported submodule onto
            # its parent by name, and the parent in sys.modules is still
            # this proxy. Materializing here would execute the real package
            # while that submodule's import is in flight: the nested package
            # __init__ then re-imports the half-initialized submodule from
            # sys.modules and the double execution corrupts package state
            # (importing dspy before a C extension that imports the
            # package's submodule directly, e.g. onnxruntime after dspy,
            # dies inside numpy with "data type 'bool' not understood").
            # Record the binding on the proxy instead; the real module's
            # own execution binds its submodules onto itself once it
            # materializes, and the stored object is the same module the
            # machinery has cached in sys.modules.
            super().__setattr__(attr, value)
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
