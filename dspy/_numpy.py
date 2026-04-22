"""Lazy numpy import helper.

``numpy`` is an optional dependency of dspy. Features that need it should call
``require_numpy()`` inside the function/method body (not at module import time),
so that ``import dspy`` continues to work for users who did not install the
``numpy`` extra.
"""
from types import ModuleType


def require_numpy() -> ModuleType:
    """Return the numpy module, or raise ImportError with install instructions."""
    try:
        import numpy
    except ImportError as e:
        raise ImportError(
            "numpy is required for this feature of dspy. "
            "Install it with `pip install dspy[numpy]` (or `pip install numpy`)."
        ) from e
    return numpy
