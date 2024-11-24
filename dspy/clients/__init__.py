from .lm import LM
from .provider import Provider, TrainingJob
from .base_lm import BaseLM, inspect_history
from .embedding import Embedder
import litellm
import os
from pathlib import Path
from litellm.caching import Cache

import ujson
import functools
from diskcache import FanoutCache

class DspyCache:
    def __init__(self, directory, size_limit):
        self.cache = FanoutCache(shards=16, timeout=2, directory=directory, size_limit=size_limit)
        self._cached_get_from_disk = functools.lru_cache(maxsize=None)(self._get_from_disk)

    def _get_from_disk(self, serialized_key):
        key = ujson.loads(serialized_key)
        return self.cache.get(key, retry=True)

    def get(self, key):
        serialized_key = ujson.dumps(key, sort_keys=True) # Serialize the dict key for the LRU cache
        return self._cached_get_from_disk(serialized_key)

    def add(self, key, value):
        self.cache.add(key, value, retry=True)
        self.get(key)

CACHEDIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
LM_CACHEDIR = os.path.join(CACHEDIR, "dspy_lm")
EMBEDDING_CACHEDIR = os.path.join(CACHEDIR, "dspy_embedding")
CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 1e10)) # 10GB limit by default

LM_CACHE = DspyCache(directory=LM_CACHEDIR, size_limit=CACHE_LIMIT)
# EMBEDDING_CACHE = DspyCache(directory=EMBEDDING_CACHEDIR, size_limit=CACHE_LIMIT)
LITELLM_GET_CACHE_KEY = litellm.Cache().get_cache_key

# This is kept for now for embeddings.
DISK_CACHEDIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
litellm.cache = Cache(disk_cache_dir=DISK_CACHEDIR, type="disk")
litellm.cache.cache.disk_cache.reset('size_limit', CACHE_LIMIT)


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
