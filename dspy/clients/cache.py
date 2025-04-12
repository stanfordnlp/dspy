import copy
import logging
import threading
from functools import wraps
from hashlib import sha256
from typing import Any, Dict, Optional

import cloudpickle
import pydantic
import ujson
from cachetools import LRUCache
from diskcache import FanoutCache

logger = logging.getLogger(__name__)


class Cache:
    """DSPy Cache

    `Cache` provides 2 levels of caching (in the given order):
        1. In-memory cache - implemented with cachetools.LRUCache
        2. On-disk cache - implemented with diskcache.FanoutCache
    """

    def __init__(
        self,
        enable_disk_cache: bool,
        enable_memory_cache: bool,
        disk_cache_dir: str,
        disk_size_limit_bytes: Optional[int] = 1024 * 1024 * 10,
        memory_max_entries: Optional[int] = 100,
    ):
        """
        Args:
            enable_disk_cache: Whether to enable on-disk cache.
            enable_memory_cache: Whether to enable in-memory cache.
            disk_cache_dir: The directory where the disk cache is stored.
            disk_size_limit_bytes: The maximum size of the disk cache (in bytes).
            memory_max_entries: The maximum size of the in-memory cache (in number of items).
        """

        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        if self.enable_memory_cache:
            self.memory_cache = LRUCache(maxsize=memory_max_entries)
        else:
            self.memory_cache = {}
        if self.enable_disk_cache:
            self.fanout_cache = FanoutCache(
                shards=16, timeout=2, directory=disk_cache_dir, size_limit=disk_size_limit_bytes
            )
        else:
            self.fanout_cache = {}

        self._lock = threading.RLock()

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return key in self.memory_cache or key in self.fanout_cache

    def cache_key(self, request: Dict[str, Any]) -> str:
        """
        Obtain a unique cache key for the given request dictionary by hashing its JSON
        representation. For request fields having types that are known to be JSON-incompatible,
        convert them to a JSON-serializable format before hashing.
        """
        params = {}
        for k, v in request.items():
            if isinstance(v, type) and issubclass(v, pydantic.BaseModel):
                params[k] = v.model_json_schema()
            elif isinstance(v, pydantic.BaseModel):
                params[k] = v.model_dump()
            else:
                params[k] = v

        return sha256(ujson.dumps(params, sort_keys=True).encode()).hexdigest()

    def get(self, request: Dict[str, Any]) -> Any:
        try:
            key = self.cache_key(request)
        except Exception:
            logger.debug(f"Failed to generate cache key for request: {request}")
            return None

        if self.enable_memory_cache and key in self.memory_cache:
            with self._lock:
                response = self.memory_cache[key]
        elif self.enable_disk_cache and key in self.fanout_cache:
            # Found on disk but not in memory cache, add to memory cache
            response = self.fanout_cache[key]
            if self.enable_memory_cache:
                with self._lock:
                    self.memory_cache[key] = response
        else:
            return None

        response = copy.deepcopy(response)
        if hasattr(response, "usage"):
            # Clear the usage data when cache is hit, because no LM call is made
            response.usage = {}
        return response

    def put(self, request: Dict[str, Any], value: Any) -> None:
        try:
            key = self.cache_key(request)
        except Exception:
            return

        if self.enable_memory_cache:
            with self._lock:
                self.memory_cache[key] = value

        if self.enable_disk_cache:
            self.fanout_cache[key] = value

    def reset_memory_cache(self) -> None:
        if not self.enable_memory_cache:
            return

        with self._lock:
            self.memory_cache.clear()

    def save_memory_cache(self, filepath: str) -> None:
        if not self.enable_memory_cache:
            return

        with self._lock:
            with open(filepath, "wb") as f:
                cloudpickle.dump(self.memory_cache, f)

    def load_memory_cache(self, filepath: str) -> None:
        if not self.enable_memory_cache:
            return

        with self._lock:
            with open(filepath, "rb") as f:
                self.memory_cache = cloudpickle.load(f)


def lm_cache(fn):
    @wraps(fn)
    def wrapper(**kwargs):
        import dspy

        cache = dspy.cache

        # Use fully qualified function name for uniqueness
        fn_identifier = f"{fn.__module__}.{fn.__qualname__}"

        # Create a modified request that includes the function identifier so that it's incorporated into the cache key.
        # Deep copy is required because litellm sometimes modifies the kwargs in place.
        modified_request = copy.deepcopy(kwargs)
        modified_request["_fn_identifier"] = fn_identifier

        # Retrieve from cache if available
        cached_result = cache.get(modified_request)

        if cached_result is not None:
            return cached_result

        # Otherwise, compute and store the result
        result = fn(**kwargs)
        cache.put(modified_request, result)
        return result

    return wrapper
