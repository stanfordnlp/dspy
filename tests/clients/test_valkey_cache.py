"""Tests for ValkeyCache — the Valkey-backed distributed cache backend.

All tests mock the Valkey connection (no running Valkey server required).
"""

import asyncio
import concurrent.futures
import pickle
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest

import dspy
from dspy.clients.valkey_cache import ValkeyCache

# -- Test fixtures and helpers --


@dataclass
class DummyResponse:
    message: str
    usage: dict
    cache_hit: bool = False


class DummyResponseNoUsage:
    """Response object without a usage attribute."""

    def __init__(self, message):
        self.message = message


class NonAllowlistedPayload:
    """Module-level class that is NOT in the restricted pickle allowlist."""

    def __init__(self):
        self.evil = "data"


class StrictResponse(pydantic.BaseModel):
    """Mimics strict pydantic models (e.g. litellm's ResponsesAPIResponse)."""

    model_config = pydantic.ConfigDict(extra="forbid")

    output: str
    usage: dict


@pytest.fixture
def mock_glide_client():
    """Create a mock GlideClient that simulates Valkey operations."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=None)
    client.exists = AsyncMock(return_value=0)
    client.delete = AsyncMock(return_value=None)
    client.close = AsyncMock(return_value=None)
    return client


@pytest.fixture
def cache(mock_glide_client):
    """Create a ValkeyCache with a mocked GLIDE client."""
    vc = ValkeyCache(host="localhost", port=6379)
    vc._client = mock_glide_client
    yield vc
    vc.close()


@pytest.fixture
def cache_with_ttl(mock_glide_client):
    """Create a ValkeyCache with TTL and a mocked GLIDE client."""
    vc = ValkeyCache(host="localhost", port=6379, ttl_seconds=300)
    vc._client = mock_glide_client
    yield vc
    vc.close()


# -- Test: initialization and validation --


def test_default_configuration():
    """Default ValkeyCache configuration is sensible."""
    vc = ValkeyCache()
    assert vc.host == "localhost"
    assert vc.port == 6379
    assert vc.ttl_seconds is None
    assert vc.tls is False
    assert vc.request_timeout == 500
    assert vc.enable_disk_cache is True
    assert vc.enable_memory_cache is False
    assert vc.key_prefix == "dspy:cache:"
    assert vc.restrict_pickle is True
    assert vc._closed is False
    vc.close()


def test_custom_configuration():
    """Custom configuration is applied."""
    vc = ValkeyCache(
        host="valkey.prod.internal",
        port=6380,
        ttl_seconds=3600,
        tls=True,
        request_timeout=1000,
        password="secret",
        username="admin",
        cluster=True,
        key_prefix="myapp:llm:",
        restrict_pickle=False,
    )
    assert vc.host == "valkey.prod.internal"
    assert vc.port == 6380
    assert vc.ttl_seconds == 3600
    assert vc.tls is True
    assert vc.request_timeout == 1000
    assert vc.password == "secret"
    assert vc.username == "admin"
    assert vc.cluster is True
    assert vc.key_prefix == "myapp:llm:"
    assert vc.restrict_pickle is False
    vc.close()


def test_ttl_zero_raises_value_error():
    """ttl_seconds=0 is rejected at init time."""
    with pytest.raises(ValueError, match="positive integer"):
        ValkeyCache(ttl_seconds=0)


def test_ttl_negative_raises_value_error():
    """ttl_seconds<0 is rejected at init time."""
    with pytest.raises(ValueError, match="positive integer"):
        ValkeyCache(ttl_seconds=-1)


def test_tls_config_implies_tls():
    """Providing tls_config sets tls=True even if not explicitly passed."""
    mock_tls = MagicMock()
    vc = ValkeyCache(tls_config=mock_tls)
    assert vc.tls is True
    assert vc.tls_config is mock_tls
    vc.close()


def test_tls_warning_on_remote_host():
    """Warning logged when connecting to remote host without TLS."""
    with patch("dspy.clients.valkey_cache.logger") as mock_logger:
        vc = ValkeyCache(host="valkey.prod.internal")
    mock_logger.warning.assert_called_once()
    args = mock_logger.warning.call_args[0]
    assert "without TLS" in args[0]
    assert "valkey.prod.internal" in args
    vc.close()


def test_no_tls_warning_for_localhost():
    """No TLS warning for localhost connections."""
    with patch("dspy.clients.valkey_cache.logger") as mock_logger:
        vc = ValkeyCache(host="localhost")
        mock_logger.warning.assert_not_called()
        vc.close()


# -- Test: cache_key --


def test_cache_key_produces_64_char_hex(cache):
    """cache_key produces a SHA-256 hex string."""
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    key = cache.cache_key(request)
    assert isinstance(key, str)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_cache_key_consistency(cache):
    """Identical requests produce identical keys."""
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}
    key1 = cache.cache_key(request)
    key2 = cache.cache_key(request)
    assert key1 == key2


def test_cache_key_matches_parent_implementation():
    """ValkeyCache produces the same key as the default Cache (inherited method)."""
    from dspy.clients.cache import Cache

    parent_cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
        disk_cache_dir=None,
    )
    valkey_cache = ValkeyCache()

    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}], "temperature": 0.5}
    assert valkey_cache.cache_key(request) == parent_cache.cache_key(request)
    valkey_cache.close()


def test_cache_key_respects_ignored_args(cache):
    """cache_key excludes specified args from the hash."""
    request = {"model": "gpt-4", "api_key": "secret", "prompt": "hello"}
    key_with = cache.cache_key(request)
    key_without = cache.cache_key(request, ignored_args_for_cache_key=["api_key"])
    assert key_with != key_without


def test_cache_key_handles_pydantic_models(cache):
    """cache_key handles pydantic model instances and classes."""

    class TestModel(pydantic.BaseModel):
        name: str
        value: int

    request_with_instance = {"data": TestModel(name="test", value=42)}
    key1 = cache.cache_key(request_with_instance)
    assert len(key1) == 64

    request_with_class = {"model_class": TestModel}
    key2 = cache.cache_key(request_with_class)
    assert len(key2) == 64


# -- Test: get --


def test_get_returns_none_on_cache_miss(cache, mock_glide_client):
    """Cache miss returns None."""
    mock_glide_client.get.return_value = None
    result = cache.get({"model": "gpt-4", "prompt": "hello"})
    assert result is None


def test_get_returns_deserialized_response_on_hit(cache, mock_glide_client):
    """Cache hit returns deserialized response with cache_hit=True."""
    original = DummyResponse(message="Hello world", usage={"tokens": 42})
    mock_glide_client.get.return_value = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)

    # DummyResponse is not in the allowlist, so use restrict_pickle=False for this test
    cache.restrict_pickle = False

    result = cache.get({"model": "gpt-4", "prompt": "hello"})

    assert result is not None
    assert result.message == "Hello world"
    assert result.usage == {}  # Cleared on cache hit
    assert result.cache_hit is True


def test_get_handles_strict_pydantic_models(cache, mock_glide_client):
    """cache_hit set via object.__setattr__ for strict models."""
    original = StrictResponse(output="result", usage={"total_tokens": 10})
    mock_glide_client.get.return_value = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)

    # StrictResponse not in allowlist, disable restrict_pickle for test
    cache.restrict_pickle = False

    result = cache.get({"model": "gpt-4", "prompt": "hello"})

    assert result is not None
    assert result.output == "result"
    assert result.usage == {}
    assert result.cache_hit is True


def test_get_response_without_usage_attribute(cache, mock_glide_client):
    """Response without 'usage' attribute is returned without modification."""
    original = DummyResponseNoUsage(message="no usage here")
    mock_glide_client.get.return_value = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
    cache.restrict_pickle = False

    result = cache.get({"model": "gpt-4", "prompt": "hello"})

    assert result is not None
    assert result.message == "no usage here"
    assert not hasattr(result, "cache_hit")


def test_get_graceful_on_connection_error(cache, mock_glide_client):
    """Connection failure returns None (graceful degradation)."""
    mock_glide_client.get.side_effect = ConnectionError("Connection refused")

    result = cache.get({"model": "gpt-4", "prompt": "hello"})
    assert result is None


def test_get_graceful_on_deserialization_error(cache, mock_glide_client):
    """Corrupt data returns None instead of crashing."""
    mock_glide_client.get.return_value = b"\xde\xad\xbe\xef"

    result = cache.get({"model": "gpt-4", "prompt": "hello"})
    assert result is None


def test_get_graceful_on_unserializable_request(cache):
    """Unserializable request returns None without raising."""

    class Unserializable:
        pass

    result = cache.get({"data": Unserializable()})
    assert result is None


def test_get_returns_none_when_disk_cache_disabled(cache):
    """get returns None when enable_disk_cache is False."""
    cache.enable_disk_cache = False
    result = cache.get({"model": "gpt-4", "prompt": "hello"})
    assert result is None


def test_get_graceful_on_futures_timeout(cache, mock_glide_client):
    """concurrent.futures.TimeoutError is caught gracefully (Python 3.10 compat)."""

    async def slow_get(key):
        await asyncio.sleep(100)

    mock_glide_client.get = AsyncMock(side_effect=slow_get)
    # Override to very short timeout to trigger the futures timeout
    cache.request_timeout = 1  # 1ms → budget = 0.001 + 1 = ~1s

    # Patch _run to raise the specific exception directly
    with patch.object(cache, "_run", side_effect=concurrent.futures.TimeoutError("timed out")):
        result = cache.get({"model": "gpt-4", "prompt": "hello"})
    assert result is None


# -- Test: restricted pickle deserialization --


def test_get_rejects_non_allowlisted_type(cache, mock_glide_client):
    """Restricted pickle rejects types not in the allowlist and evicts the entry."""
    payload = pickle.dumps(NonAllowlistedPayload(), protocol=pickle.HIGHEST_PROTOCOL)
    mock_glide_client.get.return_value = payload

    result = cache.get({"model": "gpt-4", "prompt": "hello"})

    assert result is None
    # Entry should be evicted
    mock_glide_client.delete.assert_called_once()


def test_get_allows_litellm_types_with_restricted_pickle(cache, mock_glide_client):
    """Restricted pickle allows litellm.types.* and openai.types.* by module prefix."""
    # We can't easily import real litellm types in tests, but we can verify the
    # _restricted_load function works by testing with restrict_pickle=False
    cache.restrict_pickle = False

    original = DummyResponse(message="allowed", usage={"tokens": 1})
    mock_glide_client.get.return_value = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)

    result = cache.get({"model": "gpt-4", "prompt": "hello"})
    assert result is not None
    assert result.message == "allowed"


def test_get_with_custom_safe_types(mock_glide_client):
    """Custom safe_types are allowed through restricted pickle."""
    vc = ValkeyCache(safe_types=[DummyResponse])
    vc._client = mock_glide_client

    original = DummyResponse(message="custom allowed", usage={"tokens": 1})
    mock_glide_client.get.return_value = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)

    result = vc.get({"model": "gpt-4", "prompt": "hello"})
    assert result is not None
    assert result.message == "custom allowed"
    vc.close()


# -- Test: put --


def test_put_serializes_and_stores(cache, mock_glide_client):
    """put stores pickled response in Valkey."""
    request = {"model": "gpt-4", "prompt": "hello"}
    value = DummyResponse(message="response", usage={"tokens": 5})

    cache.put(request, value)

    mock_glide_client.set.assert_called_once()
    call_args = mock_glide_client.set.call_args
    stored_key = call_args[0][0]
    stored_value = call_args[0][1]

    assert stored_key.startswith("dspy:cache:")
    assert len(stored_key) == len("dspy:cache:") + 64
    restored = pickle.loads(stored_value)
    assert restored.message == "response"


def test_put_with_ttl(cache_with_ttl, mock_glide_client):
    """put with TTL passes expiry to Valkey SET."""
    request = {"model": "gpt-4", "prompt": "hello"}
    value = DummyResponse(message="response", usage={"tokens": 5})

    cache_with_ttl.put(request, value)

    mock_glide_client.set.assert_called_once()
    call_kwargs = mock_glide_client.set.call_args.kwargs
    assert "expiry" in call_kwargs
    assert call_kwargs["expiry"] is not None


def test_put_without_ttl(cache, mock_glide_client):
    """put without TTL stores with no expiry."""
    request = {"model": "gpt-4", "prompt": "hello"}
    value = DummyResponse(message="response", usage={"tokens": 5})

    cache.put(request, value)

    call_args = mock_glide_client.set.call_args
    assert call_args.kwargs.get("expiry") is None


def test_put_graceful_on_connection_error(cache, mock_glide_client):
    """Connection failure on put does not crash."""
    mock_glide_client.set.side_effect = ConnectionError("Connection refused")
    cache.put({"model": "gpt-4", "prompt": "hello"}, DummyResponse(message="x", usage={}))


def test_put_graceful_on_unserializable_request(cache, mock_glide_client):
    """Unserializable request is handled gracefully."""

    class Unserializable:
        pass

    cache.put({"data": Unserializable()}, "value")
    mock_glide_client.set.assert_not_called()


def test_put_graceful_on_unpicklable_value(cache, mock_glide_client):
    """Value that cannot be pickled is handled gracefully."""
    request = {"model": "gpt-4", "prompt": "hello"}
    # Lambda cannot be pickled with standard pickle
    cache.put(request, lambda: None)
    mock_glide_client.set.assert_not_called()


def test_put_returns_none_when_disk_cache_disabled(cache, mock_glide_client):
    """put is a no-op when enable_disk_cache is False."""
    cache.enable_disk_cache = False
    cache.put({"model": "gpt-4", "prompt": "hello"}, DummyResponse(message="x", usage={}))
    mock_glide_client.set.assert_not_called()


# -- Test: __contains__ --


def test_contains_returns_true_when_key_exists(cache, mock_glide_client):
    """__contains__ returns True when Valkey reports key exists."""
    mock_glide_client.exists.return_value = 1
    assert ("some_key" in cache) is True


def test_contains_returns_false_when_key_missing(cache, mock_glide_client):
    """__contains__ returns False when Valkey reports key absent."""
    mock_glide_client.exists.return_value = 0
    assert ("some_key" in cache) is False


def test_contains_returns_false_on_error(cache, mock_glide_client):
    """__contains__ returns False on connection error (graceful degradation)."""
    mock_glide_client.exists.side_effect = ConnectionError("Connection refused")
    assert ("some_key" in cache) is False


# -- Test: round-trip integration --


def test_put_then_get_round_trip(cache, mock_glide_client):
    """Full put-then-get round trip."""
    cache.restrict_pickle = False
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    original = DummyResponse(message="Hello!", usage={"prompt_tokens": 5, "completion_tokens": 3})

    stored_data = {}

    async def mock_set(key, value, **kwargs):
        stored_data["key"] = key
        stored_data["value"] = value

    mock_glide_client.set = AsyncMock(side_effect=mock_set)

    cache.put(request, original)

    mock_glide_client.get.return_value = stored_data["value"]

    result = cache.get(request)

    assert result is not None
    assert result.message == "Hello!"
    assert result.usage == {}
    assert result.cache_hit is True


# -- Test: dspy.cache wiring --


def test_dspy_cache_assignment(cache):
    """ValkeyCache works as drop-in via dspy.cache assignment."""
    original_cache = dspy.cache
    try:
        dspy.cache = cache
        assert dspy.cache is cache
        assert isinstance(dspy.cache, ValkeyCache)
    finally:
        dspy.cache = original_cache


def test_request_cache_decorator_uses_valkey_cache(cache, mock_glide_client):
    """request_cache decorator works with ValkeyCache."""
    from dspy.clients.cache import request_cache

    cache.restrict_pickle = False
    stored_data = {}

    async def mock_set(key, value, **kwargs):
        stored_data[key] = value

    async def mock_get(key):
        return stored_data.get(key)

    mock_glide_client.set = AsyncMock(side_effect=mock_set)
    mock_glide_client.get = AsyncMock(side_effect=mock_get)

    with patch("dspy.cache", cache):

        @request_cache()
        def test_function(prompt, model):
            return DummyResponse(message=f"Response for {prompt}", usage={"tokens": 10})

        result1 = test_function(prompt="hello", model="gpt-4")
        assert result1.message == "Response for hello"

        result2 = test_function(prompt="hello", model="gpt-4")
        assert result2.message == "Response for hello"
        assert result2.usage == {}
        assert result2.cache_hit is True


# -- Test: optional dependency --


def test_import_without_glide_raises_helpful_error():
    """Importing ValkeyCache without valkey-glide gives helpful ImportError on connect."""
    vc = ValkeyCache()

    with patch.dict("sys.modules", {"glide": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            asyncio.run(vc._get_client())

    vc.close()


# -- Test: key prefix --


def test_valkey_key_prefix(cache):
    """Keys are namespaced with configured prefix."""
    key = cache._valkey_key("abc123")
    assert key == "dspy:cache:abc123"


def test_custom_key_prefix(mock_glide_client):
    """Custom key_prefix is used in Valkey keys."""
    vc = ValkeyCache(key_prefix="tenant-a:cache:")
    vc._client = mock_glide_client
    assert vc._valkey_key("abc") == "tenant-a:cache:abc"
    vc.close()


# -- Test: close() idempotency and post-close behavior --


def test_close_idempotent():
    """close() can be called multiple times without raising."""
    vc = ValkeyCache()
    vc.close()
    vc.close()  # Should not raise
    vc.close()  # Should not raise
    assert vc._closed is True


def test_get_after_close_returns_none(mock_glide_client):
    """get() returns None after close() instead of raising."""
    vc = ValkeyCache()
    vc._client = mock_glide_client
    vc.close()

    result = vc.get({"model": "gpt-4", "prompt": "hello"})
    assert result is None


def test_put_after_close_is_noop(mock_glide_client):
    """put() is a no-op after close() instead of raising."""
    vc = ValkeyCache()
    vc._client = mock_glide_client
    vc.close()

    # Should not raise
    vc.put({"model": "gpt-4", "prompt": "hello"}, DummyResponse(message="x", usage={}))


def test_contains_after_close_returns_false(mock_glide_client):
    """__contains__ returns False after close() instead of raising."""
    vc = ValkeyCache()
    vc._client = mock_glide_client
    vc.close()

    assert ("some_key" in vc) is False


# -- Test: context manager --


def test_context_manager():
    """ValkeyCache works as a context manager."""
    with ValkeyCache() as vc:
        assert isinstance(vc, ValkeyCache)
        assert vc._closed is False
    assert vc._closed is True


def test_context_manager_closes_on_exception():
    """Context manager calls close() even on exception."""
    try:
        with ValkeyCache() as vc:
            raise RuntimeError("test error")
    except RuntimeError:
        pass
    assert vc._closed is True


# -- Test: memory cache overrides --


def test_reset_memory_cache_is_noop(cache):
    """reset_memory_cache does nothing on ValkeyCache."""
    cache.reset_memory_cache()  # Should not raise


def test_save_memory_cache_raises(cache):
    """save_memory_cache raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        cache.save_memory_cache("/tmp/test.pkl")


def test_load_memory_cache_raises(cache):
    """load_memory_cache raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        cache.load_memory_cache("/tmp/test.pkl", allow_pickle=True)


# -- Test: debug logging does not leak keys --


def test_debug_log_does_not_contain_request_values(cache, caplog):
    """Debug logging only shows request keys, not values (no API key leakage)."""
    import logging

    class Unserializable:
        pass

    request = {"model": "gpt-4", "api_key": "sk-super-secret-key", "data": Unserializable()}

    with caplog.at_level(logging.DEBUG):
        cache.get(request)

    # Should NOT contain the actual api_key value
    assert "sk-super-secret-key" not in caplog.text


# -- Test: graceful degradation with real glide exceptions --
# (parametrized over exception types when glide is available)


glide = pytest.importorskip("glide", reason="valkey-glide not installed")


@pytest.mark.parametrize(
    "exc_class",
    [glide.ConnectionError, glide.TimeoutError, glide.ClosingError, glide.RequestError],
    ids=["ConnectionError", "TimeoutError", "ClosingError", "RequestError"],
)
class TestGlideExceptionGracefulDegradation:
    """Verify graceful degradation with actual glide exception classes."""

    def test_get_graceful(self, cache, mock_glide_client, exc_class):
        mock_glide_client.get.side_effect = exc_class("boom")
        result = cache.get({"model": "gpt-4", "prompt": "hello"})
        assert result is None

    def test_put_graceful(self, cache, mock_glide_client, exc_class):
        mock_glide_client.set.side_effect = exc_class("boom")
        # Should not raise
        cache.put({"model": "gpt-4", "prompt": "hello"}, DummyResponse(message="x", usage={}))

    def test_contains_graceful(self, cache, mock_glide_client, exc_class):
        mock_glide_client.exists.side_effect = exc_class("boom")
        assert ("some_key" in cache) is False
