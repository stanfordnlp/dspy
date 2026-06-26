"""Shared script runtime setup."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import platform

__all__ = ["configure_multiprocessing"]


def configure_multiprocessing() -> None:
    """Configure multiprocessing consistently for script entrypoints."""
    if platform.system() != "Windows":
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("fork", force=True)
    else:
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)
