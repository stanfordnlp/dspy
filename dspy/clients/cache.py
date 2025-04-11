import logging
import threading
from hashlib import sha256
from typing import Any, Dict

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
        cache_dir: str,
        enable_disk_cache: bool,
        enable_memory_cache: bool,
        disk_size_limit_bytes: int,
        memory_max_entries: int,
    ):
        """
        Args:
            cache_dir: The directory where the disk cache is stored.
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
            self.fanout_cache = FanoutCache(shards=16, timeout=2, directory=cache_dir, size_limit=disk_size_limit_bytes)
        else:
            self.fanout_cache = {}

        self._lock = threading.RLock()

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return key in self.memory_cache or key in self.fanout_cache

    def cache_key(request: Dict[str, Any]) -> str:
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

        with self.lock:
            if self.enable_memory_cache and key in self.memory_cache:
                return self.memory_cache[key]

            if self.enable_disk_cache and key in self.fanout_cache:
                # Found on disk but not in memory cache, add to memory cache
                value = self.fanout_cache[key]
                if self.enable_memory_cache:
                    self.memory_cache[key] = value
                return value

    def put(self, request: Dict[str, Any], value: Any) -> None:
        try:
            key = self.cache_key(request)
        except Exception:
            return

        if self.enable_memory_cache:
            with self.lock:
                self.memory_cache[key] = value

        if self.enable_disk_cache:
            self.fanout_cache[key] = value

    def reset_memory_cache(self) -> None:
        if not self.enable_memory_cache:
            return

        with self.lock:
            self.memory_cache.clear()


