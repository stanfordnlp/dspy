"""Lazy import helpers for optional dspy dependencies.

dspy exposes several capabilities (vector retrieval, certain optimizers, ...)
that require heavy third-party packages. Those packages are declared as
install extras (e.g. ``pip install dspy[numpy]``), and feature code paths
that need them call :func:`require_optional` inside the function body so
``import dspy`` stays cheap for users who don't opt in.
"""
import importlib
from types import ModuleType


def require_optional(module: str, *, extra: str | None = None) -> ModuleType:
    """Import ``module`` or raise :class:`ImportError` pointing to a dspy extra.

    Args:
        module: Top-level package name to import (e.g. ``"numpy"``).
        extra: Name of the dspy install extra that provides ``module``. Defaults
            to ``module``.

    Returns:
        The imported module.
    """
    extra = extra or module
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{module} is required for this feature of dspy. "
            f"Install it with `pip install dspy[{extra}]` (or `pip install {module}`)."
        ) from e
