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
    """Import ``module`` or raise :class:`ImportError`.

    Args:
        module: Top-level package name to import (e.g. ``"numpy"``).
        extra: Name of a dspy install extra that pulls in ``module``. When
            provided, the error message points users at
            ``pip install dspy[<extra>]`` in addition to ``pip install <module>``.
            Only pass a value here when the extra is actually declared in
            ``pyproject.toml``.

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        if extra:
            install_hint = f"`pip install dspy[{extra}]` (or `pip install {module}`)"
        else:
            install_hint = f"`pip install {module}`"
        raise ImportError(
            f"{module} is required for this feature of dspy. Install it with {install_hint}."
        ) from e
