import os
from pathlib import Path

_DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".dspy_cache")
DSPY_CACHEDIR = os.environ.get("DSPY_CACHEDIR") or _DEFAULT_CACHE_DIR


def create_subdir_in_cachedir(subdir: str) -> str:
    """Create a subdirectory in the DSPy cache directory."""
    subdir = os.path.join(DSPY_CACHEDIR, subdir)
    subdir = os.path.abspath(subdir)
    os.makedirs(subdir, exist_ok=True)
    return subdir
