from dspy.clients.lm import LM
from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.base_lm import BaseLM, inspect_history
from dspy.clients.embedding import Embedder
import litellm
import os
from pathlib import Path
from litellm.caching import Cache

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
DISK_CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))  # 30 GB default

# TODO: There's probably value in getting litellm to support FanoutCache and to separate the limit for
# the LM cache from the embeddings cache. Then we can lower the default 30GB limit.
litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")

if litellm.cache.cache.disk_cache.size_limit != DISK_CACHE_LIMIT:
    litellm.cache.cache.disk_cache.reset("size_limit", DISK_CACHE_LIMIT)

litellm.telemetry = False

# Turn off by default to avoid LiteLLM logging during every LM call.
litellm.suppress_debug_info = True

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    # Accessed at run time by litellm; i.e., fine to keep after import
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def enable_litellm_logging():
    litellm.suppress_debug_info = False


def disable_litellm_logging():
    litellm.suppress_debug_info = True


__all__ = [
    "BaseLM",
    "LM",
    "Provider",
    "TrainingJob",
    "inspect_history",
    "Embedder",
    "enable_litellm_logging",
    "disable_litellm_logging",
]
