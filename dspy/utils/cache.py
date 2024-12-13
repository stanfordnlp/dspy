import ujson
import pickle
import pydantic
import threading

from diskcache import FanoutCache
from cachetools import LRUCache
from hashlib import sha256
from typing import Any, Dict
from functools import wraps


class Cache:
    """
    DSPy's caching interface. It provides 2 levels of caching (in the given order):
    1. An in memory cache - cachetools' lrucache
    2. A disk based cache - diskcache's fanoutcache
    """
    
    def __init__(self, directory, disk_size_limit, mem_size_limit):
        """
        Args:
            directory: The directory where the disk cache is stored.
            disk_size_limit: The maximum size of the disk cache (in bytes).
            mem_size_limit: The maximum size of the in-memory cache (in number of items).
        """

        self.memory_cache = LRUCache(maxsize=mem_size_limit)
        self.fanout_cache = FanoutCache(shards=16, timeout=2, directory=directory, size_limit=disk_size_limit)
        self.lock = threading.RLock()

    @staticmethod
    def cache_key(request: Dict[str, Any]) -> str:
        """
        Obtain a unique cache key for the given request dictionary by hashing its JSON
        representation. For request fields having types that are known to be JSON-incompatible,
        convert them to a JSON-serializable format before hashing.
        """
        def transform_value(value):
            if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
                return value.model_json_schema() # BaseModel.schema deprecated
            elif isinstance(value, pydantic.BaseModel):
                return value.model_dump() # BaseModel.dict deprecated
            elif callable(value) and hasattr(value, "__code__") and hasattr(value.__code__, "co_code"):
                # Represent callable code objects as string
                return value.__code__.co_code.decode("utf-8")
            else:
                return value

        params = {k: transform_value(v) for k, v in request.items()}
        return sha256(ujson.dumps(params, sort_keys=True).encode()).hexdigest()
    
    def get(self, request: Dict[str, Any]) -> Any:
        try:
            key = self.cache_key(request)
        except Exception:
            return None
        
        with self.lock: # lock for thread safety (low overhead)
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

    def load(self, file_path: str):
        with open(file_path, "rb") as f:
            cache_items = pickle.load(f)
        
        with self.lock:
            for k, v in cache_items:
                self.memory_cache[k] = v

    def save(self, file_path: str) -> None:
        with self.lock: 
            cache_items = list(self.memory_cache.items())

        with open(file_path, "wb") as f:
            pickle.dump(cache_items, f)

    def reset_memory_cache(self) -> None:
        with self.lock:
            self.memory_cache.clear()


def cache_decorator(ignore=None, keep=None):
    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            import dspy
            cache = dspy.settings.cache

            # Use fully qualified function name for uniqueness
            func_identifier = f"{func.__module__}.{func.__qualname__}"

            # Create a modified request that includes the function identifier
            # so that it's incorporated into the cache key.
            modified_request = dict(kwargs)
            modified_request["_func_identifier"] = func_identifier

            for key in list(modified_request.keys()):
                if ignore and key in ignore:
                    del modified_request[key]
                if keep and key not in keep:
                    del modified_request[key]

            # Retrieve from cache if available
            cached_result = cache.get(modified_request)
            if cached_result is not None:
                return cached_result

            # Otherwise, compute and store the result
            result = func(**kwargs)
            cache.set(modified_request, result)
            return result

        return wrapper
    return decorator
