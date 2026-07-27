import logging
import os
from pathlib import Path
from typing import Any

from dspy.clients._litellm import get_litellm
from dspy.clients.base_lm import BaseLM, inspect_history
from dspy.clients.cache import Cache
from dspy.clients.embedding import Embedder
from dspy.clients.lm import LM
from dspy.clients.provider import Provider, TrainingJob

logger = logging.getLogger(__name__)

def _default_disk_cache_dir() -> str:
    """Where the on-disk cache lives, resolved without touching the filesystem.

    Evaluated at import time, so it must not raise. `Path.home()` raises
    RuntimeError where no home can be determined (WebAssembly guests, some
    containers), and `tempfile.gettempdir()` is no safer — it probes for a writable
    directory and raises FileNotFoundError when none exists. So the last resort is a
    plain relative path, resolved only if something actually opens the cache.
    """
    explicit = os.environ.get("DSPY_CACHEDIR")
    if explicit:
        return explicit
    try:
        return os.path.join(Path.home(), ".dspy_cache")
    except RuntimeError:
        return ".dspy_cache"


DISK_CACHE_DIR = _default_disk_cache_dir()
DISK_CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))  # 30 GB default


def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int = 1000000,
    restrict_pickle: bool = False,
    safe_types: list[type[Any]] | None = None,
):
    """Configure the cache for DSPy.

    Args:
        enable_disk_cache: Whether to enable on-disk cache.
        enable_memory_cache: Whether to enable in-memory cache.
        disk_cache_dir: The directory to store the on-disk cache.
        disk_size_limit_bytes: The size limit of the on-disk cache.
        memory_max_entries: The maximum number of entries in the in-memory cache. To allow the cache to grow without
                            bounds, set this parameter to `math.inf` or a similar value.
        restrict_pickle: When True, restrict pickle deserialization to a known-safe
            set of types. When False (default), use unrestricted pickle.
        safe_types: Additional types to allow when restrict_pickle is True.
    """

    DSPY_CACHE = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
        restrict_pickle=restrict_pickle,
        safe_types=safe_types,
    )

    import dspy

    # Update the reference to point to the new cache
    dspy.cache = DSPY_CACHE



def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _get_dspy_cache():
    disk_cache_dir = _default_disk_cache_dir()
    disk_cache_limit = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))

    # Both tiers off imports neither cachetools nor diskcache. That matters beyond
    # saving a little startup: diskcache pulls sqlite3, which CPython builds with a
    # reduced stdlib omit, so without a way to switch the cache off `import dspy`
    # cannot succeed there at all.
    if _env_flag("DSPY_DISABLE_CACHE"):
        return Cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
            disk_cache_dir=None,
            disk_size_limit_bytes=disk_cache_limit,
            memory_max_entries=1,
        )

    if _env_flag("DSPY_DISABLE_DISK_CACHE"):
        return Cache(
            enable_disk_cache=False,
            enable_memory_cache=True,
            disk_cache_dir=None,
            disk_size_limit_bytes=disk_cache_limit,
            memory_max_entries=1000000,
        )

    try:
        _dspy_cache = Cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
            disk_cache_dir=disk_cache_dir,
            disk_size_limit_bytes=disk_cache_limit,
            memory_max_entries=1000000,
        )
    except Exception as e:
        # If cache creation fails (e.g., in AWS Lambda), create a memory-only cache
        logger.warning("Failed to initialize disk cache, falling back to memory-only cache: %s", e)
        _dspy_cache = Cache(
            enable_disk_cache=False,
            enable_memory_cache=True,
            disk_cache_dir=disk_cache_dir,
            disk_size_limit_bytes=disk_cache_limit,
            memory_max_entries=1000000,
        )
    return _dspy_cache


DSPY_CACHE = _get_dspy_cache()


def configure_litellm_logging(level: str = "ERROR"):
    """Configure LiteLLM logging to the specified level."""
    # Litellm uses a global logger called `verbose_logger` to control all loggings.
    litellm = get_litellm(feature="LiteLLM logging")
    verbose_logger = litellm._logging.verbose_logger

    numeric_logging_level = getattr(logging, level)

    verbose_logger.setLevel(numeric_logging_level)
    for h in verbose_logger.handlers:
        h.setLevel(numeric_logging_level)


def enable_litellm_logging():
    litellm = get_litellm(feature="LiteLLM logging")
    litellm.suppress_debug_info = False
    litellm._dspy_logging_configured = True
    configure_litellm_logging("DEBUG")


def disable_litellm_logging():
    litellm = get_litellm(feature="LiteLLM logging")
    litellm.suppress_debug_info = True
    litellm._dspy_logging_configured = True
    configure_litellm_logging("ERROR")

__all__ = [
    "BaseLM",
    "LM",
    "Provider",
    "TrainingJob",
    "inspect_history",
    "Embedder",
    "enable_litellm_logging",
    "disable_litellm_logging",
    "configure_cache",
]
