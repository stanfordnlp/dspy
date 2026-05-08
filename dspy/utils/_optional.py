"""Lazy import helpers for optional dspy dependencies.

dspy exposes several capabilities (vector retrieval, certain optimizers, ...)
that require heavy third-party packages. Those packages are declared as
install extras in ``pyproject.toml`` (e.g. ``pip install dspy[numpy]``), and
feature code paths that need them call :func:`require_optional` inside the
function body so ``import dspy`` stays cheap for users who don't opt in.

``_DSPY_EXTRAS`` maps each importable module name to the dspy extra that
provides it; keep it in sync with ``[project.optional-dependencies]`` in
``pyproject.toml``.
"""
import importlib
from types import ModuleType

_DSPY_EXTRAS: dict[str, str] = {
    "numpy": "numpy",
}


def require_optional(module: str) -> ModuleType:
    """Import ``module`` or raise :class:`ImportError` with an install hint.

    If ``module`` is registered in :data:`_DSPY_EXTRAS`, the error message
    additionally suggests ``pip install dspy[<extra>]``.

    Args:
        module: Top-level package name to import (e.g. ``"numpy"``).

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        extra = _DSPY_EXTRAS.get(module)
        if extra:
            hint = f"`pip install dspy[{extra}]` (or `pip install {module}`)"
        else:
            hint = f"`pip install {module}`"
        raise ImportError(
            f"{module} is required for this feature of dspy. Install it with {hint}."
        ) from e
