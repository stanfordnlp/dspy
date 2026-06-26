"""Shared script runtime setup."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import platform
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

__all__ = ["configure_multiprocessing", "load_env_file"]


def load_env_file(env_file: str | Path = DEFAULT_ENV_FILE) -> Path | None:
    """Load package-local environment variables if the file exists."""
    env_path = Path(env_file)
    if not env_path.exists():
        return None
    load_dotenv(env_path, override=False)
    return env_path


def configure_multiprocessing() -> None:
    """Configure multiprocessing consistently for script entrypoints."""
    if platform.system() != "Windows":
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("fork", force=True)
    else:
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)
