import copy
import dataclasses
import inspect
import logging
import os
import threading
import warnings
from functools import wraps
from hashlib import sha256
from typing import Any

import cloudpickle
import orjson
import pydantic
from cachetools import LRUCache
from diskcache import FanoutCache

from dspy.clients.disk_serialization import (
    SAFE_TAG,
    DeserializationError,
    LegacyFormatError,
    decode,
    default_allowed_types,
    make_safe_disk,
)

logger = logging.getLogger(__name__)


def _transform_value(value):
    if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
        return value.model_json_schema()
    elif isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _transform_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    elif callable(value):
        # Try to get the source code of the callable if available
        try:
            # For regular functions, we can get the source code
            return f"<callable_source:{inspect.getsource(value)}>"
        except (TypeError, OSError):
            # For lambda functions or other callables where source isn't available,
            # use a string representation
            return f"<callable:{value.__name__ if hasattr(value, '__name__') else 'lambda'}>"
    elif isinstance(value, dict):
        return {k: _transform_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_transform_value(v) for v in value]
    elif isinstance(value, (set, frozenset)):
        transformed_values = [_transform_value(v) for v in value]
        return sorted(
            transformed_values,
            key=lambda item: orjson.dumps(item, option=orjson.OPT_SORT_KEYS),
        )
    else:
        return value


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
        disk_size_limit_bytes: int | None = 1024 * 1024 * 10,
        memory_max_entries: int = 1000000,
        use_pickle: bool = True,
        safe_types: list[type[Any]] | None = None,
    ):
        """
        Args:
            enable_disk_cache: Whether to enable on-disk cache.
            enable_memory_cache: Whether to enable in-memory cache.
            disk_cache_dir: The directory where the disk cache is stored.
            disk_size_limit_bytes: The maximum size of the disk cache (in bytes).
            memory_max_entries: The maximum size of the in-memory cache (in number of items).
            use_pickle: When True (default), use pickle serialization for disk cache.
                When False, use the safe serializer.
            safe_types: Optional top-level pydantic model or dataclass types to
                register for safe disk-cache serialization.
        """

        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        self.use_pickle = use_pickle
        self.disk_cache_dir = os.fspath(disk_cache_dir)
        self._allowed_types = default_allowed_types()
        for cls in safe_types or []:
            self._allowed_types.add((cls.__module__, cls.__name__))
        if self.enable_memory_cache:
            if memory_max_entries is None:
                raise ValueError("`memory_max_entries` cannot be None. Use `math.inf` if you need an unbounded cache.")
            elif memory_max_entries <= 0:
                raise ValueError(f"`memory_max_entries` must be a positive number, but received {memory_max_entries}")
            self.memory_cache = LRUCache(maxsize=memory_max_entries)
        else:
            self.memory_cache = {}
        if self.enable_disk_cache:
            fanout_kwargs = dict(
                directory=self.disk_cache_dir,
                shards=16,
                timeout=10,
                size_limit=disk_size_limit_bytes if disk_size_limit_bytes is not None else 2**40,
                eviction_policy="none" if disk_size_limit_bytes is None else "least-recently-stored",
            )
            if not use_pickle:
                fanout_kwargs["disk"] = make_safe_disk(self._allowed_types)
            self.disk_cache = FanoutCache(**fanout_kwargs)
        else:
            self.disk_cache = {}

        self._lock = threading.RLock()

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return key in self.memory_cache or key in self.disk_cache

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> str:
        """
        Obtain a unique cache key for the given request dictionary by hashing its JSON
        representation. For request fields having types that are known to be JSON-incompatible,
        convert them to a JSON-serializable format before hashing.
        """

        ignored_args_for_cache_key = ignored_args_for_cache_key or []
        params = {k: _transform_value(v) for k, v in request.items() if k not in ignored_args_for_cache_key}
        return sha256(orjson.dumps(params, option=orjson.OPT_SORT_KEYS)).hexdigest()

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> Any:
        if not self.enable_memory_cache and not self.enable_disk_cache:
            return None

        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request: %s", request)
            return None

        if self.enable_memory_cache:
            with self._lock:
                response = self.memory_cache.get(key)
            if response is not None:
                return self._prepare_cached_response(response)

        if self.enable_disk_cache:
            try:
                result = self.disk_cache.get(key, tag=self.use_pickle)
            except LegacyFormatError:
                warnings.warn(
                    "Existing disk cache entry could not be deserialized and will be skipped. "
                    "This is expected when switching from pickle to safe serialization mode. "
                    "Affected entries will be re-computed and stored in the new format.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            except DeserializationError:
                logger.debug("Failed to deserialize disk cache entry %s", key)
                return None

            if self.use_pickle:
                response, tag = result
                if response is None:
                    return None
                if tag == SAFE_TAG:
                    try:
                        response = decode(response, allowed=self._allowed_types)
                    except DeserializationError:
                        logger.debug("Failed to decode safe-mode cache entry %s", key)
                        return None
            else:
                response = result
                if response is None:
                    return None

            if self.enable_memory_cache:
                with self._lock:
                    self.memory_cache[key] = response
            return self._prepare_cached_response(response)

        return None

    def _prepare_cached_response(self, response):
        response = copy.deepcopy(response)
        if hasattr(response, "usage"):
            # Clear the usage data when cache is hit, because no LM call is made
            response.usage = {}
            response.cache_hit = True
        return response

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: list[str] | None = None,
        enable_memory_cache: bool = True,
    ) -> None:
        enable_memory_cache = self.enable_memory_cache and enable_memory_cache

        # Early return to avoid computing cache key if both memory and disk cache are disabled
        if not enable_memory_cache and not self.enable_disk_cache:
            return

        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request: %s", request)
            return

        if enable_memory_cache:
            with self._lock:
                self.memory_cache[key] = value

        if self.enable_disk_cache:
            try:
                self.disk_cache.set(key, value, tag=None if self.use_pickle else SAFE_TAG)
            except TypeError as e:
                warnings.warn(f"Skipping disk cache write: {e}", UserWarning, stacklevel=2)
            except Exception as e:
                logger.debug("Failed to put value in disk cache: %s, %s", value, e)

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

    def load_memory_cache(self, filepath: str, allow_pickle: bool = False) -> None:
        if not allow_pickle:
            raise ValueError(
                "Loading untrusted .pkl files can run arbitrary code, which may be dangerous. "
                "Set `allow_pickle=True` to load if you are running in a trusted environment and the file is from a trusted source."
            )

        if not self.enable_memory_cache:
            return

        with self._lock:
            with open(filepath, "rb") as f:
                self.memory_cache = cloudpickle.load(f)


def request_cache(
    cache_arg_name: str | None = None,
    ignored_args_for_cache_key: list[str] | None = None,
    enable_memory_cache: bool = True,
    *,  # everything after this is keyword-only
    maxsize: int | None = None,  # legacy / no-op
):
    """
    Decorator for applying caching to a function based on the request argument.

    Args:
        cache_arg_name: The name of the argument that contains the request. If not provided, the entire kwargs is used
            as the request.
        ignored_args_for_cache_key: A list of arguments to ignore when computing the cache key from the request.
        enable_memory_cache: Whether to enable in-memory cache at call time. If False, the memory cache will not be
            written to on new data.
    """
    ignored_args_for_cache_key = ignored_args_for_cache_key or ["api_key", "api_base", "base_url"]
    # Deprecation notice
    if maxsize is not None:
        logger.warning(
            "[DEPRECATION] `maxsize` is deprecated and no longer does anything; "
            "the cache is now handled internally by `dspy.cache`. "
            "This parameter will be removed in a future release.",
        )

    def decorator(fn):
        @wraps(fn)
        def build_cache_request(args, kwargs):
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

            return modified_request

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            import dspy

            cache = dspy.cache
            modified_request = build_cache_request(args, kwargs)

            # Retrieve from cache if available
            cached_result = cache.get(modified_request, ignored_args_for_cache_key)
            if cached_result is not None:
                return cached_result

            # Otherwise, compute and store the result
            # Make a copy of the original request in case it's modified in place, e.g., deleting some fields
            original_request = copy.deepcopy(modified_request)
            result = fn(*args, **kwargs)
            # `enable_memory_cache` can be provided at call time to avoid indefinite growth.
            cache.put(original_request, result, ignored_args_for_cache_key, enable_memory_cache)

            return result

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            import dspy

            cache = dspy.cache
            modified_request = build_cache_request(args, kwargs)

            # Retrieve from cache if available
            cached_result = cache.get(modified_request, ignored_args_for_cache_key)
            if cached_result is not None:
                return cached_result

            # Otherwise, compute and store the result
            # Make a copy of the original request in case it's modified in place, e.g., deleting some fields
            original_request = copy.deepcopy(modified_request)
            result = await fn(*args, **kwargs)
            cache.put(original_request, result, ignored_args_for_cache_key, enable_memory_cache)

            return result

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
