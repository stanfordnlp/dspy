import pickle
import threading
from diskcache import FanoutCache
from cachetools import LRUCache
import ujson
import pydantic
from hashlib import sha256
from typing import Any, Dict
from functools import wraps

class Cache:
    """
    dspy's caching interface. It provides 2 levels of caching (in the given order): 
    1. An in memory cache - cachetools' lrucache 
    2. A disk based cache - diskcache's fanoutcache

    The cache is threadsafe

    Args: 
        mem_size_limit: The maximum size of the cache. If unspecified, no max size is enforced (cache is unbounded).
        disk_size_limit: 
    """
    def __init__(self, directory, disk_size_limit, mem_size_limit):
        self.memory_cache = LRUCache(maxsize=mem_size_limit)
        self.fanout_cache = FanoutCache(shards=16, timeout=2, directory=directory, size_limit=disk_size_limit)
        self.lock = threading.RLock()

    @staticmethod
    def cache_key(request: Dict[str, Any]) -> str:
        """
        Obtain a unique cache key for the given request dictionary by hashing its JSON
        representation. For request fields having types that are known to be JSON-incompatible,
        convert them to a JSON-serializable format before hashing.

        Note: Values that cannot be converted to JSON should *not* be ignored / discarded, since
        that would potentially lead to cache collisions. For example, consider request A
        containing only JSON-convertible values and request B containing the same JSON-convertible
        values in addition to one unconvertible value. Discarding the unconvertible value would
        lead to a cache collision between requests A and B, even though they are semantically
        different.
        """

        def transform_value(value):
            if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
                return value.model_json_schema() # BaseModel.schema deprecated
            elif isinstance(value, pydantic.BaseModel):
                return value.model_dump() # BaseModel.dict deprecated
            elif callable(value) and hasattr(value, "__code__") and hasattr(value.__code__, "co_code"):
                return value.__code__.co_code.decode("utf-8")
            else:
                # Note: We don't attempt to compute a hash of the value, since the default
                # implementation of hash() is id(), which may collide if the same memory address
                # is reused for different objects at different times
                return value

        params = {k: transform_value(v) for k, v in request.items()}
        return sha256(ujson.dumps(params, sort_keys=True).encode()).hexdigest()
    
    def get(self, request: Dict[str, Any]) -> Any:
        try:
            key = self.cache_key(request)
        except Exception:
            return None
        
        with self.lock: # TODO: Do we need this lock for reads? LRUCache is ambiguous!
            if key in self.memory_cache:  
                return self.memory_cache[key]
            
            if key in self.fanout_cache:
                # found on disk but not in memory, add to memory cache
                value = self.fanout_cache[key]
                self.memory_cache[key] = value
                return value

    def set(self, request: Dict[str, Any], value: Any) -> None:
        try: 
            key = self.cache_key(request)
        except Exception:
            return None
        
        with self.lock:
            self.memory_cache[key] = value
            self.fanout_cache[key] = value

    def load(self, file_path, maxsize:float=float("inf")) -> "LRUCache":
        with open(file_path, "rb") as f:
            cache_items = pickle.load(f)
        
        with self.lock:
            # self.memory_cache.clear()
            for k,v in cache_items:
                self.memory_cache[k] = v

    def save(self, file_path: str) -> None:
        with self.lock: 
            cache_items = list(self.memory_cache.items())

        with open(file_path, "wb") as f:
            pickle.dump(cache_items, f)

    def reset_memory_cache(self) -> None:
        with self.lock:
            self.memory_cache.clear()

def cache_decorator():
    import dspy
    cache = dspy.settings.cache

    # TODO: FIXME: The name of the decorated function should be part of the cache key

    def decorator(func):
        @wraps(func)
        def wrapper(request: dict, *args, **kwargs):
            cached_result = cache.get(request)
            if cached_result is not None:
                return cached_result
            
            result = func(request, *args, **kwargs)
            cache.set(request, result)
            return result
        
        return wrapper
    
    return decorator

