import os
from pathlib import Path


def default_cache_dir() -> str:
    """Resolve the cache directory without touching the filesystem.

    Evaluated at import time, so it must not raise. See the matching helper in
    `dspy.clients`, deliberately duplicated rather than shared: importing
    `dspy.utils` from `dspy.clients` would close an import cycle.
    """
    explicit = os.environ.get("DSPY_CACHEDIR")
    if explicit:
        return explicit
    try:
        return os.path.join(Path.home(), ".dspy_cache")
    except RuntimeError:
        return ".dspy_cache"


DSPY_CACHEDIR = default_cache_dir()


def create_subdir_in_cachedir(subdir: str) -> str:
    """Create a subdirectory in the DSPy cache directory."""
    subdir = os.path.join(DSPY_CACHEDIR, subdir)
    subdir = os.path.abspath(subdir)
    os.makedirs(subdir, exist_ok=True)
    return subdir
