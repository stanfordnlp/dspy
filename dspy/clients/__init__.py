import logging
import os
from pathlib import Path

import litellm

from dspy.clients.base_lm import BaseLM, inspect_history
from dspy.clients.cache import Cache
from dspy.clients.embedding import Embedder
from dspy.clients.lm import LM
from dspy.clients.provider import Provider, TrainingJob

logger = logging.getLogger(__name__)

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
DISK_CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))  # 30 GB default
def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int | None = 1000000,
):
    """Configure the cache for DSPy.

    Args:
        enable_disk_cache: Whether to enable on-disk cache.
        enable_memory_cache: Whether to enable in-memory cache.
        disk_cache_dir: The directory to store the on-disk cache.
        disk_size_limit_bytes: The size limit of the on-disk cache.
        memory_max_entries: The maximum number of entries in the in-memory cache.
    """

    DSPY_CACHE = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
    )

    import dspy
    # Update the reference to point to the new cache
    dspy.cache = DSPY_CACHE


litellm.telemetry = False
litellm.cache = None  # By default we disable LiteLLM cache and use DSPy on-disk cache.

def _get_dspy_cache():
    disk_cache_dir = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
    disk_cache_limit = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))

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

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    # Accessed at run time by litellm; i.e., fine to keep after import
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def configure_litellm_logging(level: str = "ERROR"):
    """Configure LiteLLM logging to the specified level."""
    # Litellm uses a global logger called `verbose_logger` to control all loggings.
    from litellm._logging import verbose_logger

    numeric_logging_level = getattr(logging, level)

    verbose_logger.setLevel(numeric_logging_level)
    for h in verbose_logger.handlers:
        h.setLevel(numeric_logging_level)


def enable_litellm_logging():
    litellm.suppress_debug_info = False
    configure_litellm_logging("DEBUG")


def disable_litellm_logging():
    litellm.suppress_debug_info = True
    configure_litellm_logging("ERROR")


# By default, we disable LiteLLM logging for clean logging
disable_litellm_logging()

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
