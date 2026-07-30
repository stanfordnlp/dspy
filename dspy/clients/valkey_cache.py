"""Valkey-backed distributed cache backend for DSPy.

Replaces DSPy's default local diskcache + cachetools with a Valkey server,
enabling shared LLM response caching across multiple workers and environments.

Requires the optional ``valkey-glide`` package::

    pip install valkey-glide

Usage::

    import dspy
    from dspy.clients.valkey_cache import ValkeyCache

    dspy.cache = ValkeyCache(host="localhost", port=6379)

Note on trust model: cached values are deserialized with ``pickle.loads``.
This matches DSPy's default diskcache behavior. Only connect to trusted
Valkey instances — do not point this at untrusted servers.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import pickle
import threading
from hashlib import sha256
from typing import Any

import orjson

from dspy.clients.cache import Cache, _transform_value

logger = logging.getLogger(__name__)

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
    )
except ImportError:
    _VALKEY_ERRORS = (OSError, TimeoutError, ConnectionError)


class ValkeyCache(Cache):
    """DSPy Cache backed by Valkey via valkey-glide.

    Implements the same interface as the default ``Cache`` class (``cache_key``,
    ``get``, ``put``, ``__contains__``) but stores entries in a Valkey server
    instead of local disk/memory.

    Uses a background daemon thread with its own asyncio event loop to bridge
    DSPy's synchronous cache calls to the async valkey-glide client.

    Supports both standalone and cluster Valkey deployments. For cluster mode,
    set ``cluster=True`` — valkey-glide handles topology discovery, slot routing,
    and failover automatically.
    """

    KEY_PREFIX = "dspy:cache:"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        ttl_seconds: int | None = None,
        tls: bool = False,
        request_timeout: int = 500,
        password: str | None = None,
        username: str | None = None,
        cluster: bool = False,
    ):
        """Initialize a Valkey-backed cache.

        Args:
            host: Valkey server hostname or IP.
            port: Valkey server port.
            ttl_seconds: Optional TTL for cache entries in seconds. None means no expiry.
                Use 0 for no expiry explicitly (equivalent to None).
            tls: Whether to use TLS for the connection.
            request_timeout: Timeout for individual Valkey commands in milliseconds.
            password: Password for Valkey AUTH. Required for password-protected instances.
            username: Username for Valkey ACL authentication (Valkey 6+).
            cluster: Whether to connect in cluster mode. When True, uses GlideClusterClient
                which handles topology discovery and multi-node routing.
        """
        # Deliberately skip super().__init__() — we don't need diskcache or cachetools.
        self.host = host
        self.port = port
        self.ttl_seconds = ttl_seconds
        self.tls = tls
        self.request_timeout = request_timeout
        self.password = password
        self.username = username
        self.cluster = cluster

        # Flags inspected by DSPy's request_cache decorator.
        self.enable_disk_cache = True  # Valkey acts as our "disk" tier
        self.enable_memory_cache = False  # No local memory layer

        # Satisfy duck-typing expectations of parent class attributes.
        self.memory_cache: dict = {}
        self.disk_cache: dict = {}

        # Background event loop for async valkey-glide operations.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="valkey-cache-io")
        self._thread.start()
        self._client = None
        self._client_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro):
        """Submit a coroutine to the background loop and block for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=5.0)

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
        return f"{self.KEY_PREFIX}{cache_key}"

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> str:
        """Compute a deterministic cache key for the request.

        Produces identical output to the parent class implementation — SHA-256
        of orjson-serialized request dict with ignored args filtered out and
        non-JSON-serializable values transformed.
        """
        ignored_args_for_cache_key = ignored_args_for_cache_key or []
        params = {k: _transform_value(v) for k, v in request.items() if k not in ignored_args_for_cache_key}
        return sha256(orjson.dumps(params, option=orjson.OPT_SORT_KEYS)).hexdigest()

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> Any:
        """Retrieve a cached response from Valkey.

        Returns None on cache miss or any connection/deserialization error
        (graceful degradation).
        """
        if not self.enable_disk_cache:
            return None

        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request: %s", request)
            return None

        try:
            raw = self._run(self._async_get(key))
        except _VALKEY_ERRORS:
            logger.warning("Valkey GET failed for key %s; treating as cache miss", key[:16])
            return None

        if raw is None:
            return None

        try:
            response = pickle.loads(raw)
        except Exception:
            logger.debug("Failed to deserialize cached value for key %s", key[:16])
            return None

        return self._prepare_cached_response(response)

    async def _async_get(self, key: str) -> bytes | None:
        client = await self._get_client()
        return await client.get(self._valkey_key(key))

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
        if not self.enable_disk_cache:
            return

        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug("Failed to generate cache key for request: %s", request)
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
        try:
            return self._run(self._async_exists(key))
        except _VALKEY_ERRORS:
            return False

    async def _async_exists(self, key: str) -> bool:
        client = await self._get_client()
        count = await client.exists([self._valkey_key(key)])
        return count > 0

    # ------------------------------------------------------------------
    # Response preparation (matches parent behavior)
    # ------------------------------------------------------------------

    def _prepare_cached_response(self, response):
        """Deep-copy response, clear usage, and mark as cache hit.

        Matches the parent class contract: clears ``usage`` and sets
        ``cache_hit = True`` using ``object.__setattr__`` for compatibility
        with strict pydantic models.
        """
        response = copy.deepcopy(response)
        if hasattr(response, "usage"):
            response.usage = {}
            object.__setattr__(response, "cache_hit", True)
        return response

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the Valkey connection and stop the background event loop.

        Safe to call multiple times. After close(), further get/put calls
        will fail gracefully (return None / no-op).
        """
        if self._client:
            try:
                self._run(self._client.close())
            except Exception:
                pass
            self._client = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        self._loop.close()

    def __del__(self):
        """Best-effort cleanup on garbage collection."""
        try:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=1)
            if not self._loop.is_closed():
                self._loop.close()
        except Exception:
            pass
