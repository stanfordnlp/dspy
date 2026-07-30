"""Tests for ValkeyCache — the Valkey-backed distributed cache backend.

All tests mock the Valkey connection (no running Valkey server required).
"""

import asyncio
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
    client.close = AsyncMock(return_value=None)
    return client


@pytest.fixture
def cache(mock_glide_client):
    """Create a ValkeyCache with a mocked GLIDE client."""
    vc = ValkeyCache(host="localhost", port=6379)
    # Inject mock client directly, bypassing actual connection
    vc._client = mock_glide_client
    return vc


@pytest.fixture
def cache_with_ttl(mock_glide_client):
    """Create a ValkeyCache with TTL and a mocked GLIDE client."""
    vc = ValkeyCache(host="localhost", port=6379, ttl_seconds=300)
    vc._client = mock_glide_client
    return vc


# -- Test: cache_key --


def test_cache_key_produces_64_char_hex(cache):
    """TDD 2.2: cache_key produces a SHA-256 hex string."""
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    key = cache.cache_key(request)
    assert isinstance(key, str)
    assert len(key) == 64
    # All hex characters
    assert all(c in "0123456789abcdef" for c in key)


def test_cache_key_consistency(cache):
    """TDD 2.2: Identical requests produce identical keys."""
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}
    key1 = cache.cache_key(request)
    key2 = cache.cache_key(request)
    assert key1 == key2


def test_cache_key_matches_parent_implementation():
    """TDD 2.2: ValkeyCache produces the same key as the default Cache."""
    from dspy.clients.cache import Cache

    parent_cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
        disk_cache_dir=None,
    )
    valkey_cache = ValkeyCache.__new__(ValkeyCache)
    # Minimal init for cache_key to work
    valkey_cache.enable_disk_cache = True
    valkey_cache.enable_memory_cache = False
    valkey_cache.memory_cache = {}
    valkey_cache.disk_cache = {}

    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}], "temperature": 0.5}
    assert valkey_cache.cache_key(request) == parent_cache.cache_key(request)


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
    """TDD 1.3: Cache miss returns None."""
    mock_glide_client.get.return_value = None
    result = cache.get({"model": "gpt-4", "prompt": "hello"})
    assert result is None


def test_get_returns_deserialized_response_on_hit(cache, mock_glide_client):
    """TDD 1.2 / 2.1: Cache hit returns deserialized response with cache_hit=True."""
    original = DummyResponse(message="Hello world", usage={"tokens": 42})
    mock_glide_client.get.return_value = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)

    result = cache.get({"model": "gpt-4", "prompt": "hello"})

    assert result is not None
    assert result.message == "Hello world"
    assert result.usage == {}  # Cleared on cache hit
    assert result.cache_hit is True


def test_get_handles_strict_pydantic_models(cache, mock_glide_client):
    """TDD 2.1: cache_hit set via object.__setattr__ for strict models."""
    original = StrictResponse(output="result", usage={"total_tokens": 10})
    mock_glide_client.get.return_value = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)

    result = cache.get({"model": "gpt-4", "prompt": "hello"})

    assert result is not None
    assert result.output == "result"
    assert result.usage == {}
    assert result.cache_hit is True


def test_get_graceful_on_connection_error(cache, mock_glide_client):
    """TDD 4.1: Connection failure returns None (graceful degradation)."""
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


# -- Test: put --


def test_put_serializes_and_stores(cache, mock_glide_client):
    """TDD 1.1: put stores pickled response in Valkey."""
    request = {"model": "gpt-4", "prompt": "hello"}
    value = DummyResponse(message="response", usage={"tokens": 5})

    cache.put(request, value)

    mock_glide_client.set.assert_called_once()
    call_args = mock_glide_client.set.call_args
    stored_key = call_args[0][0]
    stored_value = call_args[0][1]

    assert stored_key.startswith("dspy:cache:")
    assert len(stored_key) == len("dspy:cache:") + 64
    # Verify the stored bytes can be deserialized
    restored = pickle.loads(stored_value)
    assert restored.message == "response"


def test_put_with_ttl(cache_with_ttl, mock_glide_client):
    """TDD 3.1: put with TTL passes expiry to Valkey SET."""
    request = {"model": "gpt-4", "prompt": "hello"}
    value = DummyResponse(message="response", usage={"tokens": 5})

    cache_with_ttl.put(request, value)

    mock_glide_client.set.assert_called_once()
    # Verify expiry kwarg was passed (non-None)
    call_kwargs = mock_glide_client.set.call_args.kwargs
    assert "expiry" in call_kwargs
    assert call_kwargs["expiry"] is not None


def test_put_without_ttl(cache, mock_glide_client):
    """TDD 3.2: put without TTL stores with no expiry."""
    request = {"model": "gpt-4", "prompt": "hello"}
    value = DummyResponse(message="response", usage={"tokens": 5})

    cache.put(request, value)

    # set called with just key and value (no expiry kwarg)
    call_args = mock_glide_client.set.call_args
    assert call_args.kwargs.get("expiry") is None


def test_put_graceful_on_connection_error(cache, mock_glide_client):
    """TDD 4.1: Connection failure on put does not crash."""
    mock_glide_client.set.side_effect = ConnectionError("Connection refused")

    # Should not raise
    cache.put({"model": "gpt-4", "prompt": "hello"}, DummyResponse(message="x", usage={}))


def test_put_graceful_on_unserializable_request(cache, mock_glide_client):
    """Unserializable request is handled gracefully."""

    class Unserializable:
        pass

    # Should not raise
    cache.put({"data": Unserializable()}, "value")
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
    """TDD 1.2 / 2.1: Full put-then-get round trip."""
    request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    original = DummyResponse(message="Hello!", usage={"prompt_tokens": 5, "completion_tokens": 3})

    # Capture what put stores
    stored_data = {}

    async def mock_set(key, value, **kwargs):
        stored_data["key"] = key
        stored_data["value"] = value

    mock_glide_client.set = AsyncMock(side_effect=mock_set)

    cache.put(request, original)

    # Now simulate get returning what was stored
    mock_glide_client.get.return_value = stored_data["value"]

    result = cache.get(request)

    assert result is not None
    assert result.message == "Hello!"
    assert result.usage == {}
    assert result.cache_hit is True


# -- Test: dspy.cache wiring --


def test_dspy_cache_assignment(cache):
    """TDD 1.1: ValkeyCache works as drop-in via dspy.cache assignment."""
    original_cache = dspy.cache
    try:
        dspy.cache = cache
        assert dspy.cache is cache
        assert isinstance(dspy.cache, ValkeyCache)
    finally:
        dspy.cache = original_cache


def test_request_cache_decorator_uses_valkey_cache(cache, mock_glide_client):
    """TDD 1.1: request_cache decorator works with ValkeyCache."""
    from dspy.clients.cache import request_cache

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

        # First call — cache miss, computes result
        result1 = test_function(prompt="hello", model="gpt-4")
        assert result1.message == "Response for hello"

        # Second call — cache hit
        result2 = test_function(prompt="hello", model="gpt-4")
        assert result2.message == "Response for hello"
        assert result2.usage == {}
        assert result2.cache_hit is True


# -- Test: optional dependency --


def test_import_without_glide_raises_helpful_error():
    """TDD 5.1: Importing ValkeyCache without valkey-glide gives helpful ImportError."""
    # ValkeyCache itself imports fine (deferred glide import).
    # The error only surfaces when _get_client is called.
    vc = ValkeyCache.__new__(ValkeyCache)
    vc.host = "localhost"
    vc.port = 6379
    vc.request_timeout = 500
    vc.tls = False
    vc.password = None
    vc.username = None
    vc.cluster = False
    vc._client = None
    vc._loop = asyncio.new_event_loop()
    vc._client_lock = asyncio.Lock()
    vc._thread = MagicMock()

    with patch.dict("sys.modules", {"glide": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            asyncio.run(vc._get_client())

    vc._loop.close()


# -- Test: key prefix --


def test_valkey_key_prefix(cache):
    """Keys are namespaced with dspy:cache: prefix."""
    key = cache._valkey_key("abc123")
    assert key == "dspy:cache:abc123"


# -- Test: configuration --


def test_default_configuration():
    """Default ValkeyCache configuration is sensible."""
    vc = ValkeyCache.__new__(ValkeyCache)
    ValkeyCache.__init__(vc)
    assert vc.host == "localhost"
    assert vc.port == 6379
    assert vc.ttl_seconds is None
    assert vc.tls is False
    assert vc.request_timeout == 500
    assert vc.enable_disk_cache is True
    assert vc.enable_memory_cache is False
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
    )
    assert vc.host == "valkey.prod.internal"
    assert vc.port == 6380
    assert vc.ttl_seconds == 3600
    assert vc.tls is True
    assert vc.request_timeout == 1000
    assert vc.password == "secret"
    assert vc.username == "admin"
    assert vc.cluster is True
    vc.close()


def test_ttl_zero_treated_as_expiry():
    """TDD 3.1: ttl_seconds=0 should still set expiry (not treated as None)."""
    vc = ValkeyCache(ttl_seconds=0)
    # 0 is not None, so expiry path should be taken
    assert vc.ttl_seconds is not None
    assert vc.ttl_seconds == 0
    vc.close()
