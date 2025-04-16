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
        memory_max_entries: Optional[int] = 1000000,
        ignored_args_for_cache_key: Optional[list[str]] = None,
    ):
        """
        Args:
            enable_disk_cache: Whether to enable on-disk cache.
            enable_memory_cache: Whether to enable in-memory cache.
            disk_cache_dir: The directory where the disk cache is stored.
            disk_size_limit_bytes: The maximum size of the disk cache (in bytes).
            memory_max_entries: The maximum size of the in-memory cache (in number of items).
            ignored_args_for_cache_key: A list of arguments to ignore when computing the cache key from the request.
        """

        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        if self.enable_memory_cache:
            self.memory_cache = LRUCache(maxsize=memory_max_entries)
        else:
            self.memory_cache = {}
        if self.enable_disk_cache:
            self.disk_cache = FanoutCache(
                shards=16,
                timeout=10,
                directory=disk_cache_dir,
                size_limit=disk_size_limit_bytes,
            )
        else:
            self.disk_cache = {}

        self.ignored_args_for_cache_key = ignored_args_for_cache_key or []

        self._lock = threading.RLock()

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return key in self.memory_cache or key in self.disk_cache

    def cache_key(self, request: Dict[str, Any]) -> str:
        """
        Obtain a unique cache key for the given request dictionary by hashing its JSON
        representation. For request fields having types that are known to be JSON-incompatible,
        convert them to a JSON-serializable format before hashing.
        """

        def transform_value(value):
            if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
                return value.model_json_schema()
            elif isinstance(value, pydantic.BaseModel):
                return value.model_dump()
            elif callable(value):
                # Try to get the source code of the callable if available
                import inspect

                try:
                    # For regular functions, we can get the source code
                    return f"<callable_source:{inspect.getsource(value)}>"
                except (TypeError, OSError, IOError):
                    # For lambda functions or other callables where source isn't available,
                    # use a string representation
                    return f"<callable:{value.__name__ if hasattr(value, '__name__') else 'lambda'}>"
            elif isinstance(value, dict):
                return {k: transform_value(v) for k, v in value.items()}
            else:
                return value

        params = {k: transform_value(v) for k, v in request.items() if k not in self.ignored_args_for_cache_key}
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
        elif self.enable_disk_cache and key in self.disk_cache:
            # Found on disk but not in memory cache, add to memory cache
            response = self.disk_cache[key]
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
            logger.debug(f"Failed to generate cache key for request: {request}")
            return

        if self.enable_memory_cache:
            with self._lock:
                self.memory_cache[key] = value

        if self.enable_disk_cache:
            try:
                self.disk_cache[key] = value
            except Exception as e:
                # Disk cache writing can fail for different reasons, e.g. disk full or the `value` is not picklable.
                logger.debug(f"Failed to put value in disk cache: {value}, {e}")

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


def request_cache(cache_arg_name: Optional[str] = None, ignored_args_for_cache_key: Optional[list[str]] = None):
    """Decorator for applying caching to a function based on the request argument.

    Args:
        cache_arg_name: The name of the argument that contains the request. If not provided, the entire kwargs is used
            as the request.
        ignored_args_for_cache_key: A list of arguments to ignore when computing the cache key from the request.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            import dspy

            cache = dspy.cache
            original_ignored_args_for_cache_key = cache.ignored_args_for_cache_key
            cache.ignored_args_for_cache_key = ignored_args_for_cache_key or []

            # Use fully qualified function name for uniqueness
            fn_identifier = f"{fn.__module__}.{fn.__qualname__}"

            # Create a modified request that includes the function identifier so that it's incorporated into the cache
            # key. Deep copy is required because litellm sometimes modifies the kwargs in place.
            if cache_arg_name:
                # When `cache_arg_name` is provided, use the value of the argument with this name as the request for
                # caching.
                modified_request = copy.deepcopy(kwargs[cache_arg_name])
            else:
                # When `cache_arg_name` is not provided, use the entire kwargs as the request for caching.
                modified_request = copy.deepcopy(kwargs)
                for i, arg in enumerate(args):
                    modified_request[f"positional_arg_{i}"] = arg
            modified_request["_fn_identifier"] = fn_identifier

            # Retrieve from cache if available
            cached_result = cache.get(modified_request)

            if cached_result is not None:
                return cached_result

            # Otherwise, compute and store the result
            result = fn(*args, **kwargs)
            cache.put(modified_request, result)

            cache.ignored_args_for_cache_key = original_ignored_args_for_cache_key
            return result

        return wrapper

    return decorator
