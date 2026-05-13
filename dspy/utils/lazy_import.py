"""Lazy-import helpers for optional dependencies.

Optional deps must be importable lazily so that `import dspy` succeeds even
when they are absent. Call sites get a module-level binding that defers the
real import until first attribute access:

    from dspy.utils.lazy_import import require

    np = require("numpy")          # zero cost -- no import happens here
    np.array([1, 2, 3])            # numpy is loaded on first use

If the package is not installed, the first attribute access raises
`ImportError` with an install hint.

The lazy-load machinery is vendored from *lazy_loader* (BSD-3, Scientific
Python team) and uses `importlib.util.LazyLoader` under the hood.
"""

import functools
import importlib
import importlib.metadata
import importlib.util
import inspect
import sys
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
}


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

    **Installed** -- returns a `LazyLoader`-wrapped module. The real import
    happens on first attribute access; afterwards the object is a plain module.

    **Not installed** -- returns a `_MissingModule` stub. The first attribute
    access raises `ImportError` with a `pip install dspy[…]` hint and the
    file/line where `require()` was originally called.

    Args:
        module: Dotted module path (e.g. `"numpy"`).
        extra: Name of the dspy extra that provides this dep.
        feature: Label shown in the error (e.g. `"dspy.Embeddings"`).
    """
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

    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module] = mod
    loader.exec_module(mod)
    return mod
