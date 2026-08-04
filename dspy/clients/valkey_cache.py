"""Valkey-backed distributed cache backend for DSPy.

Replaces DSPy's default local diskcache + cachetools with a Valkey server,
enabling shared LLM response caching across multiple workers and environments.

Requires the optional ``valkey-glide`` package::

    pip install dspy[valkey]

Usage::

    import dspy
    from dspy.clients.valkey_cache import ValkeyCache

    dspy.cache = ValkeyCache(host="localhost", port=6379)

    # With context manager for guaranteed cleanup:
    with ValkeyCache(host="localhost", port=6379) as cache:
        dspy.cache = cache
        # ... use dspy ...

Trust model: cached values are deserialized with restricted pickle by default.
Only types from litellm.types.* and openai.types.* are allowed. Pass
``restrict_pickle=False`` to disable this hardening (not recommended for
shared Valkey instances).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging
import pickle
import threading
from typing import Any

from dspy.clients.cache import Cache
from dspy.clients.disk_serialization import DeserializationError, _restricted_load

logger = logging.getLogger(__name__)

# Save a reference to the builtin TimeoutError before glide's import may shadow it.
_BUILTIN_TIMEOUT_ERROR = TimeoutError

# GLIDE-specific exceptions for network I/O error handling.
# Resolved at import time if glide is available; falls back to builtins otherwise.
try:
    from glide import ClosingError, ConnectionError, RequestError, TimeoutError

    _VALKEY_ERRORS: tuple[type[BaseException], ...] = (
        ClosingError,
        ConnectionError,
        RequestError,
        TimeoutError,
        OSError,
        concurrent.futures.TimeoutError,
    )
except ImportError:
    _VALKEY_ERRORS = (OSError, concurrent.futures.TimeoutError)


class ValkeyCache(Cache):
    """DSPy Cache backed by Valkey via valkey-glide.

    Implements the same interface as the default ``Cache`` class (``get``,
    ``put``, ``__contains__``) but stores entries in a Valkey server
    instead of local disk/memory.

    Uses a background daemon thread with its own asyncio event loop to bridge
    DSPy's synchronous cache calls to the async valkey-glide client.

    Supports both standalone and cluster Valkey deployments. For cluster mode,
    set ``cluster=True`` — valkey-glide handles topology discovery, slot routing,
    and failover automatically.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        ttl_seconds: int | None = None,
        tls: bool = False,
        tls_config: Any | None = None,
        request_timeout: int = 500,
        password: str | None = None,
        username: str | None = None,
        cluster: bool = False,
        key_prefix: str = "dspy:cache:",
        restrict_pickle: bool = True,
        safe_types: list[type] | None = None,
    ):
        """Initialize a Valkey-backed cache.

        Args:
            host: Valkey server hostname or IP.
            port: Valkey server port.
            ttl_seconds: Optional TTL for cache entries in seconds. None means no expiry.
                Must be a positive integer if provided.
            tls: Whether to use TLS for the connection.
            tls_config: Optional TLS configuration object (e.g.,
                ``glide.TlsAdvancedConfiguration(...)``). When provided, implies ``tls=True``.
                Use for custom CA certs, mTLS client certificates, or other advanced TLS settings.
            request_timeout: Timeout for individual Valkey commands in milliseconds.
            password: Password for Valkey AUTH. Required for password-protected instances.
            username: Username for Valkey ACL authentication (Valkey 6+).
            cluster: Whether to connect in cluster mode. When True, uses GlideClusterClient
                which handles topology discovery and multi-node routing.
            key_prefix: Prefix for all cache keys in Valkey. Use distinct prefixes for
                tenant isolation on shared Valkey instances.
            restrict_pickle: When True (default), restrict deserialization to known-safe
                types (litellm/openai response models). Prevents arbitrary code execution
                from poisoned cache entries on shared Valkey instances.
            safe_types: Additional types to allow when restrict_pickle is True.
        """
        # Deliberately skip super().__init__() — we don't need diskcache or cachetools.
        # Validate ttl_seconds
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer; use None for no expiry")

        self.host = host
        self.port = port
        self.ttl_seconds = ttl_seconds
        self.tls = tls or (tls_config is not None)
        self.tls_config = tls_config
        self.request_timeout = request_timeout
        self.password = password
        self.username = username
        self.cluster = cluster
        self.key_prefix = key_prefix
        self.restrict_pickle = restrict_pickle

        # Build restricted-pickle allowlist.
        for t in safe_types or []:
            if not isinstance(t, type):
                raise TypeError(f"safe_types entries must be types, got {t!r}")
        self._allowed = frozenset((cls.__module__, cls.__qualname__) for cls in (safe_types or []))

        # TLS warning for non-localhost connections.
        if not self.tls and host not in ("localhost", "127.0.0.1", "::1"):
            logger.warning(
                "ValkeyCache connecting to remote host %s without TLS. "
                "Prompts, completions, and the AUTH password will be sent in plaintext. "
                "Pass tls=True.",
                host,
            )

        # Flags inspected by DSPy's request_cache decorator.
        self.enable_disk_cache = True  # Valkey acts as our "disk" tier
        self.enable_memory_cache = False  # No local memory layer

        # Satisfy duck-typing expectations of parent class attributes.
        self.memory_cache: dict = {}
        self.disk_cache: dict = {}

        # Provide _lock for safety — parent code uses it under enable_memory_cache guard,
        # but having it avoids AttributeError if assumptions change.
        self._lock = threading.RLock()

        # Background event loop for async valkey-glide operations.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="valkey-cache-io")
        self._thread.start()
        self._client = None
        self._client_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro):
        """Submit a coroutine to the background loop and block for the result.

        Timeout is derived from request_timeout (ms) + 1s buffer for connection overhead.
        On Python 3.10, concurrent.futures.TimeoutError is a separate class from
        builtins.TimeoutError — we normalize it to builtins.TimeoutError for consistent
        handling by _VALKEY_ERRORS.
        """
        if self._closed:
            raise OSError("ValkeyCache is closed")
        budget = self.request_timeout / 1000 + 1.0
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=budget)
        except concurrent.futures.TimeoutError as e:
            raise _BUILTIN_TIMEOUT_ERROR(str(e)) from e

    async def _get_client(self):
        """Lazily create and return the GLIDE client (thread-safe via asyncio.Lock)."""
        async with self._client_lock:
            if self._client is None:
                from glide import (
                    GlideClient,
                    GlideClientConfiguration,
                    GlideClusterClient,
                    GlideClusterClientConfiguration,
                    NodeAddress,
                    ServerCredentials,
                )

                addresses = [NodeAddress(self.host, self.port)]

                # Build credentials if auth is configured.
                credentials = None
                if self.password is not None:
                    credentials = ServerCredentials(password=self.password, username=self.username or "")

                if self.cluster:
                    config = GlideClusterClientConfiguration(
                        addresses=addresses,
                        request_timeout=self.request_timeout,
                        use_tls=self.tls,
                        credentials=credentials,
                    )
                    self._client = await GlideClusterClient.create(config)
                else:
                    config = GlideClientConfiguration(
                        addresses=addresses,
                        request_timeout=self.request_timeout,
                        use_tls=self.tls,
                        credentials=credentials,
                    )
                    self._client = await GlideClient.create(config)
        return self._client

    def _valkey_key(self, cache_key: str) -> str:
        """Build the namespaced Valkey key from a cache key hash."""
        return f"{self.key_prefix}{cache_key}"

    def _deserialize(self, raw: bytes) -> Any:
        """Deserialize cached bytes with restricted or unrestricted pickle."""
        if self.restrict_pickle:
            return _restricted_load(io.BytesIO(raw), self._allowed)
        return pickle.loads(raw)

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> Any:
        """Retrieve a cached response from Valkey.

        Returns None on cache miss or any connection/deserialization error
        (graceful degradation).
        """
        if self._closed or not self.enable_disk_cache:
            return None

        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request with keys: %s", list(request.keys()))
            return None

        try:
            raw = self._run(self._async_get(key))
        except _VALKEY_ERRORS:
            logger.warning("Valkey GET failed for key %s; treating as cache miss", key[:16])
            return None

        if raw is None:
            return None

        try:
            response = self._deserialize(raw)
        except DeserializationError:
            logger.warning("Rejected non-allowlisted cached value for key %s; evicting", key[:16])
            try:
                self._run(self._async_delete(key))
            except _VALKEY_ERRORS:
                pass
            return None
        except Exception:
            logger.debug("Failed to deserialize cached value for key %s", key[:16])
            return None

        return self._prepare_cached_response(response)

    async def _async_get(self, key: str) -> bytes | None:
        client = await self._get_client()
        return await client.get(self._valkey_key(key))

    async def _async_delete(self, key: str) -> None:
        """Delete a cache entry from Valkey (for evicting poisoned entries)."""
        client = await self._get_client()
        await client.delete([self._valkey_key(key)])

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: list[str] | None = None,
        enable_memory_cache: bool = True,
    ) -> None:
        """Store a response in Valkey.

        Silently fails on connection errors (graceful degradation).
        """
        if self._closed or not self.enable_disk_cache:
            return

        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request with keys: %s", list(request.keys()))
            return

        try:
            raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logger.debug("Failed to serialize value for caching: %s", type(value))
            return

        try:
            self._run(self._async_put(key, raw))
        except _VALKEY_ERRORS:
            logger.warning("Valkey SET failed for key %s; skipping cache write", key[:16])

    async def _async_put(self, key: str, value: bytes) -> None:
        client = await self._get_client()
        vkey = self._valkey_key(key)

        if self.ttl_seconds is not None:
            from glide import ExpirySet, ExpiryType

            expiry = ExpirySet(expiry_type=ExpiryType.SEC, value=self.ttl_seconds)
            await client.set(vkey, value, expiry=expiry)
        else:
            await client.set(vkey, value)

    def __contains__(self, key: str) -> bool:
        """Check if a cache key exists in Valkey."""
        if self._closed:
            return False
        try:
            return self._run(self._async_exists(key))
        except _VALKEY_ERRORS:
            return False

    async def _async_exists(self, key: str) -> bool:
        client = await self._get_client()
        count = await client.exists([self._valkey_key(key)])
        return count > 0

    # ------------------------------------------------------------------
    # Response preparation (optimized for network deserialization)
    # ------------------------------------------------------------------

    def _prepare_cached_response(self, response):
        """Mark response as cache hit and clear usage.

        Skips deepcopy because pickle.loads() already returns a fresh,
        unshared object — no aliasing risk from the Valkey byte stream.
        """
        if hasattr(response, "usage"):
            response.usage = {}
            object.__setattr__(response, "cache_hit", True)
        return response

    # ------------------------------------------------------------------
    # Memory cache overrides (not supported on ValkeyCache)
    # ------------------------------------------------------------------

    def reset_memory_cache(self) -> None:
        """No-op: ValkeyCache does not use a local memory tier."""

    def save_memory_cache(self, filepath: str) -> None:
        """Not supported: ValkeyCache does not use a local memory tier."""
        raise NotImplementedError("ValkeyCache does not support memory cache serialization")

    def load_memory_cache(self, filepath: str, allow_pickle: bool = False) -> None:
        """Not supported: ValkeyCache does not use a local memory tier."""
        raise NotImplementedError("ValkeyCache does not support memory cache loading")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the Valkey connection and stop the background event loop.

        Safe to call multiple times. After close(), further get/put calls
        will degrade gracefully (return None / no-op).
        """
        if self._closed:
            return
        self._closed = True

        if self._client:
            try:
                future = asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
                future.result(timeout=2)
            except Exception:
                pass
            self._client = None

        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            self._loop.close()

    def __enter__(self):
        """Support use as context manager."""
        return self

    def __exit__(self, *exc):
        """Close on context manager exit."""
        self.close()

    def __del__(self):
        """Best-effort cleanup on garbage collection."""
        try:
            if not self._closed:
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    self._thread.join(timeout=1)
                if not self._loop.is_closed():
                    self._loop.close()
        except Exception:
            pass
