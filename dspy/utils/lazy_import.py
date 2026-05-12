"""Lazy-import helpers for optional dependencies.

Dspy ships in two flavors with different hard-dependency sets (`dspy` and
`dspy-runtime`). Optional deps must be importable lazily so that `import dspy`
succeeds even when they are absent, and call sites must raise a clear,
actionable ImportError when the dep really is needed.
"""

import functools
import importlib
import importlib.util
from typing import Any

_INSTALL_HINTS: dict[str, str] = {
    "optuna": "optuna",
    "mcp": "mcp",
    "langchain_core": "langchain",
    "weaviate": "weaviate",
    "anthropic": "anthropic",
    "numpy": "numpy",
}


@functools.cache
def is_available(module: str) -> bool:
    """Return True if ``module`` can be imported, without importing it.

    Uses ``importlib.util.find_spec`` so calling this does not execute the
    module's top-level code. Safe for cheap branching ("if the optional dep
    is installed, register the hook; otherwise skip").
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, *, extra: str | None = None, feature: str | None = None) -> Any:
    """Import a module by dotted path; raise a friendly ImportError if missing.

    Use at call sites where an optional dependency is needed to perform an action.

    Args:
        module: Dotted module path (e.g. ``"litellm"`` or ``"gepa.core.adapter"``).
            The top-level segment is shown to the user.
        extra: Name of the dspy extra that pulls in this dep. Defaults to the
            entry in ``_INSTALL_HINTS`` for the top-level module, falling back
            to the top-level module name.
        feature: Short feature label included in the error (e.g. ``"dspy.LM"``).
            Defaults to ``"this feature"``.

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        top = module.split(".", 1)[0]
        feat = feature or "this feature"
        ext = extra or _INSTALL_HINTS.get(top, top)
        raise ImportError(
            f"{top} is required to use {feat}. "
            f"Install with `pip install dspy[{ext}]` or `pip install {top}`."
        ) from e


def optional(module: str, attr: str | None = None, default: Any = None) -> Any:
    """Try to import a module (and optionally one attribute). Return ``default`` if missing.

    Use at module load time when a class needs to inherit from a base provided by
    an optional dep: returning a sentinel (typically ``object``) lets the class be
    defined even when the dep is absent. Gate actual use behind ``require()``.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return default
    if attr is None:
        return mod
    return getattr(mod, attr, default)
