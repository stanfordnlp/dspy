import logging
import os
from pathlib import Path
from typing import Optional

# Set environment variables before importing litellm
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["OPENAI_LOG"] = "ERROR"

import litellm
from litellm.caching.caching import Cache as LitellmCache

def _configure_litellm_logging(level: str = "ERROR"):
    """Configure LiteLLM logging to the specified level."""
    # Update environment variables
    os.environ["LITELLM_LOG"] = level
    os.environ["OPENAI_LOG"] = level
    
    # Cover both capitalization variants used by LiteLLM
    logger_names = [
        "LiteLLM",
        "LiteLLM.utils",
        "LiteLLM.proxy.utils",
        "litellm",
        "litellm.utils",
        "litellm.proxy.utils",
    ]
    _level = getattr(logging, level)
    for logger_name in logger_names:
        lg = logging.getLogger(logger_name)
        lg.setLevel(_level)
        lg.propagate = False
        # Remove all existing handlers or force them to the desired level
        for h in lg.handlers[:]:
            h.setLevel(_level)
        # Ensure there is at least a NullHandler to swallow logs
        if not lg.handlers:
            lg.addHandler(logging.NullHandler())

# Immediately disable LiteLLM logging after import
_configure_litellm_logging("ERROR")

from dspy.clients.base_lm import BaseLM, inspect_history
from dspy.clients.cache import Cache
from dspy.clients.embedding import Embedder
from dspy.clients.lm import LM
from dspy.clients.provider import Provider, TrainingJob

logger = logging.getLogger(__name__)

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
DISK_CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))  # 30 GB default


def _litellm_track_cache_hit_callback(kwargs, completion_response, start_time, end_time):
    # Access the cache_hit information
    completion_response.cache_hit = kwargs.get("cache_hit", False)


litellm.success_callback = [_litellm_track_cache_hit_callback]


def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int | None = 1000000,
    enable_litellm_cache: bool = False,
):
    """Configure the cache for DSPy.

    Args:
        enable_disk_cache: Whether to enable on-disk cache.
        enable_memory_cache: Whether to enable in-memory cache.
        disk_cache_dir: The directory to store the on-disk cache.
        disk_size_limit_bytes: The size limit of the on-disk cache.
        memory_max_entries: The maximum number of entries in the in-memory cache.
        enable_litellm_cache: Whether to enable LiteLLM cache.
    """
    if enable_disk_cache and enable_litellm_cache:
        raise ValueError(
            "Cannot enable both LiteLLM and DSPy on-disk cache, please set at most one of `enable_disk_cache` or "
            "`enable_litellm_cache` to True."
        )

    if enable_litellm_cache:
        try:
            litellm.cache = LitellmCache(disk_cache_dir=DISK_CACHE_DIR, type="disk")

            if litellm.cache.cache.disk_cache.size_limit != DISK_CACHE_LIMIT:
                litellm.cache.cache.disk_cache.reset("size_limit", DISK_CACHE_LIMIT)
        except Exception as e:
            # It's possible that users don't have the write permissions to the cache directory.
            # In that case, we'll just disable the cache.
            logger.warning("Failed to initialize LiteLLM cache: %s", e)
            litellm.cache = None
    else:
        litellm.cache = None

    import dspy

    dspy.cache = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
    )


litellm.telemetry = False
litellm.cache = None  # By default we disable litellm cache and use DSPy on-disk cache.

DSPY_CACHE = Cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=DISK_CACHE_DIR,
    disk_size_limit_bytes=DISK_CACHE_LIMIT,
    memory_max_entries=1000000,
)

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    # Accessed at run time by litellm; i.e., fine to keep after import
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def enable_litellm_logging():
    litellm.suppress_debug_info = False
    _configure_litellm_logging("INFO")
    # Remove environment variables to allow logging
    if "LITELLM_LOG" in os.environ:
        del os.environ["LITELLM_LOG"]
    if "OPENAI_LOG" in os.environ:
        del os.environ["OPENAI_LOG"]


def disable_litellm_logging():
    litellm.suppress_debug_info = True
    _configure_litellm_logging("ERROR")


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
