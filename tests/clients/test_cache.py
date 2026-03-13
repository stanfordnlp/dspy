import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from unittest.mock import patch

import orjson
import pydantic
import pytest
from cachetools import LRUCache

from dspy.clients.cache import Cache
from dspy.clients.sqlite_cache import SQLiteCache


class CacheValidationModel(pydantic.BaseModel):
    name: str
    value: int


@pytest.fixture
def cache_config(tmp_path):
    """Default cache configuration."""
    return {
        "enable_disk_cache": True,
        "enable_memory_cache": True,
        "disk_cache_dir": str(tmp_path),
        "disk_size_limit_bytes": 1024 * 1024,  # 1MB
        "memory_max_entries": 100,
    }


@pytest.fixture
def cache(cache_config):
    """Create a cache instance with the default configuration."""
    return Cache(**cache_config)


def test_initialization(tmp_path):
    """Test different cache initialization configurations."""
    # Test memory-only cache
    memory_cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir="",
        disk_size_limit_bytes=0,
        memory_max_entries=50,
    )
    assert isinstance(memory_cache.memory_cache, LRUCache)
    assert memory_cache.memory_cache.maxsize == 50
    assert memory_cache.disk_cache == {}

    # Test disk-only cache
    disk_cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024,
        memory_max_entries=0,
    )
    assert isinstance(disk_cache.disk_cache, SQLiteCache)
    assert disk_cache.memory_cache == {}

    # Test disabled cache
    disabled_cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
        disk_cache_dir="",
        disk_size_limit_bytes=0,
        memory_max_entries=0,
    )
    assert disabled_cache.memory_cache == {}
    assert disabled_cache.disk_cache == {}


def test_invalid_cache_initialization():
    with pytest.raises(ValueError, match=r"`memory_max_entries` must be a positive number, but received -1"):
        Cache(
            enable_disk_cache=False,
            enable_memory_cache=True,
            disk_cache_dir="",
            disk_size_limit_bytes=0,
            memory_max_entries=-1,
        )
    with pytest.raises(
        ValueError, match=r"`memory_max_entries` cannot be None. Use `math.inf` if you need an unbounded cache."
    ):
        Cache(
            enable_disk_cache=False,
            enable_memory_cache=True,
            disk_cache_dir="",
            disk_size_limit_bytes=0,
            memory_max_entries=None,
        )


def test_cache_key_generation(cache):
    """Test cache key generation with different types of inputs."""
    # Test with simple dictionary
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini", "temperature": 0.7}
    key = cache.cache_key(request)
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hash is 64 characters

    # Test with pydantic model
    class TestModel(pydantic.BaseModel):
        name: str
        value: int

    model = TestModel(name="test", value=42)
    request_with_model = {"data": model}
    key_with_model = cache.cache_key(request_with_model)
    assert isinstance(key_with_model, str)

    # Test with pydantic model class
    request_with_model_class = {"model_class": TestModel}
    key_with_model_class = cache.cache_key(request_with_model_class)
    assert isinstance(key_with_model_class, str)


def test_put_and_get(cache):
    """Test putting and getting from cache."""
    # Test putting and getting from memory cache
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini", "temperature": 0.7}
    value = {"message": "This is a test response", "usage": {"prompt_tokens": 10, "completion_tokens": 20}}

    cache.put(request, value)
    result = cache.get(request)

    assert result == value

    # Test with disk cache
    # First, clear memory cache to ensure we're using disk cache
    cache.reset_memory_cache()

    # Get from disk cache
    result_from_disk = cache.get(request)
    assert result_from_disk == value

    # Verify it was also added back to memory cache
    assert cache.cache_key(request) in cache.memory_cache


def test_cache_miss(cache):
    """Test getting a non-existent key."""
    request = {"prompt": "Non-existent", "model": "gpt-4"}
    result = cache.get(request)
    assert result is None


def test_cache_key_error_handling(cache):
    """Test error handling for unserializable objects."""

    # Test with a request that can't be serialized to JSON
    class UnserializableObject:
        pass

    request = {"data": UnserializableObject()}

    # Should not raise an exception
    result = cache.get(request)
    assert result is None

    # Should not raise an exception
    cache.put(request, "value")


def test_full_cache_put_get_cycle_with_model_response(cache):
    from litellm import ModelResponse

    response = ModelResponse(
        id="test-123",
        choices=[{"message": {"content": "cached response"}, "index": 0, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    request = {"model": "openai/gpt-5-nano", "prompt": "test"}
    cache.put(request, response)
    cache.reset_memory_cache()

    result = cache.get(request)
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "cached response"
    assert result.usage == {}
    assert result.cache_hit is True


def test_disk_cache_write_unsupported_value_warns_and_skips_disk(cache):
    class NotSerializable:
        pass

    request = {"model": "test", "prompt": "test_unserializable_value"}
    with pytest.warns(UserWarning, match="Skipping disk cache write"):
        cache.put(request, NotSerializable())

    assert isinstance(cache.get(request), NotSerializable)

    cache.reset_memory_cache()
    assert cache.get(request) is None


def test_cache_contains_checks_disk(cache):
    request = {"prompt": "test"}
    key = cache.cache_key(request)

    assert key not in cache
    cache.put(request, "value")
    assert key in cache
    cache.reset_memory_cache()
    assert key in cache


def test_disk_deserialization_error_returns_none(cache):
    """If a disk cache entry can't be deserialized (e.g. class was removed),
    Cache.get() should return None, not crash."""
    request = {"model": "test", "prompt": "broken_entry"}
    key = cache.cache_key(request)
    envelope = orjson.dumps(
        {
            "_data": {
                "__dspy_cache_type__": "pydantic",
                "__dspy_cache_module__": "nonexistent.module",
                "__dspy_cache_qualname__": "FakeClass",
                "__dspy_cache_data__": {"x": 1},
            }
        }
    )
    with cache.disk_cache._lock:
        conn = cache.disk_cache._get_conn()
        conn.execute(
            "INSERT INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
            (key, envelope, len(envelope), 0.0),
        )
        conn.commit()
    cache.reset_memory_cache()
    assert cache.get(request) is None


def test_disk_corrupt_json_returns_none(cache):
    request = {"model": "test", "prompt": "corrupt_json"}
    key = cache.cache_key(request)

    with cache.disk_cache._lock:
        conn = cache.disk_cache._get_conn()
        conn.execute(
            "INSERT INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
            (key, b"{not valid json", len(b"{not valid json"), 0.0),
        )
        conn.commit()

    cache.reset_memory_cache()
    assert cache.get(request) is None


def test_disk_pydantic_validation_error_returns_none(cache):
    request = {"model": "test", "prompt": "invalid_pydantic"}
    key = cache.cache_key(request)
    envelope = orjson.dumps(
        {
            "_data": {
                "__dspy_cache_type__": "pydantic",
                "__dspy_cache_module__": CacheValidationModel.__module__,
                "__dspy_cache_qualname__": CacheValidationModel.__qualname__,
                "__dspy_cache_data__": {"name": "missing_required_value"},
            }
        }
    )

    with cache.disk_cache._lock:
        conn = cache.disk_cache._get_conn()
        conn.execute(
            "INSERT INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
            (key, envelope, len(envelope), 0.0),
        )
        conn.commit()

    cache.reset_memory_cache()
    assert cache.get(request) is None


def test_reset_memory_cache(cache):
    """Test resetting memory cache."""
    # Add some items to the memory cache
    requests = [{"prompt": f"Hello {i}", "model": "openai/gpt-4o-mini"} for i in range(5)]
    for i, req in enumerate(requests):
        cache.put(req, f"Response {i}")

    # Verify items are in memory cache
    for req in requests:
        key = cache.cache_key(req)
        assert key in cache.memory_cache

    # Reset memory cache
    cache.reset_memory_cache()

    # Verify memory cache is empty
    assert len(cache.memory_cache) == 0

    # But disk cache still has the items
    for req in requests:
        result = cache.get(req)
        assert result is not None


def test_get_returns_value_when_memory_cache_is_reset_concurrently(tmp_path):
    contains_started = threading.Event()
    allow_contains_return = threading.Event()
    result = {}

    class PausingMemoryCache:
        def __init__(self):
            self._data = {}

        def __contains__(self, key):
            present = key in self._data
            contains_started.set()
            assert allow_contains_return.wait(timeout=2)
            return present

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def clear(self):
            self._data.clear()

    cache = Cache(
        enable_memory_cache=True,
        enable_disk_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=0,
        memory_max_entries=10,
    )
    cache.memory_cache = PausingMemoryCache()
    request = {"prompt": "race", "model": "openai/gpt-5-nano"}
    cache.put(request, "value")

    def reader():
        try:
            result["value"] = cache.get(request)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    read_thread = threading.Thread(target=reader)
    reset_thread = threading.Thread(target=cache.reset_memory_cache)

    read_thread.start()
    assert contains_started.wait(timeout=2)
    reset_thread.start()
    allow_contains_return.set()

    read_thread.join(timeout=2)
    reset_thread.join(timeout=2)

    assert not read_thread.is_alive()
    assert not reset_thread.is_alive()
    assert result == {"value": "value"}


def test_save_and_load_memory_cache(cache, tmp_path):
    """Test saving and loading memory cache."""
    # Add some items to the memory cache
    requests = [{"prompt": f"Hello {i}", "model": "openai/gpt-4o-mini"} for i in range(5)]
    for i, req in enumerate(requests):
        cache.put(req, f"Response {i}")

    # Save memory cache to a temporary file
    temp_cache_file = tmp_path / "memory_cache.pkl"
    cache.save_memory_cache(str(temp_cache_file))

    # Create a new cache instance with disk cache disabled
    new_cache = Cache(
        enable_memory_cache=True,
        enable_disk_cache=False,
        disk_cache_dir=tmp_path / "disk_cache",
        disk_size_limit_bytes=0,
        memory_max_entries=100,
    )

    # Load the memory cache without allowing pickle (default)
    with pytest.raises(ValueError):
        new_cache.load_memory_cache(str(temp_cache_file))

    # Load the memory cache with allow_pickle=True
    new_cache.load_memory_cache(str(temp_cache_file), allow_pickle=True)

    # Verify items are in the new memory cache
    for req in requests:
        result = new_cache.get(req)
        assert result is not None
        assert result == f"Response {requests.index(req)}"


def test_request_cache_decorator(cache):
    """Test the lm_cache decorator."""
    from dspy.clients.cache import request_cache

    # Mock the dspy.cache attribute
    with patch("dspy.cache", cache):
        # Define a test function
        @request_cache()
        def test_function(prompt, model):
            return f"Response for {prompt} with {model}"

        # First call should compute the result
        result1 = test_function(prompt="Hello", model="openai/gpt-4o-mini")
        assert result1 == "Response for Hello with openai/gpt-4o-mini"

        # Second call with same arguments should use cache
        with patch.object(cache, "get") as mock_get:
            mock_get.return_value = "Cached response"
            result2 = test_function(prompt="Hello", model="openai/gpt-4o-mini")
            assert result2 == "Cached response"
            mock_get.assert_called_once()

        # Call with different arguments should compute again
        result3 = test_function(prompt="Different", model="openai/gpt-4o-mini")
        assert result3 == "Response for Different with openai/gpt-4o-mini"


def test_request_cache_decorator_with_ignored_args_for_cache_key(cache):
    """Test the request_cache decorator with ignored_args_for_cache_key."""
    from dspy.clients.cache import request_cache

    # Mock the dspy.cache attribute
    with patch("dspy.cache", cache):
        # Define a test function
        @request_cache(ignored_args_for_cache_key=["model"])
        def test_function1(prompt, model):
            return f"Response for {prompt} with {model}"

        @request_cache()
        def test_function2(prompt, model):
            return f"Response for {prompt} with {model}"

        # First call should compute the result
        result1 = test_function1(prompt="Hello", model="openai/gpt-4o-mini")
        result2 = test_function1(prompt="Hello", model="openai/gpt-4o")

        # Because model arg is ignored, the second call should return the same result as the first
        assert result1 == result2

        result3 = test_function2(prompt="Hello", model="openai/gpt-4o-mini")
        result4 = test_function2(prompt="Hello", model="openai/gpt-4o")

        # Because model arg is not ignored, the second call should return a different result
        assert result3 != result4


@pytest.mark.asyncio
async def test_request_cache_decorator_async(cache):
    """Test the request_cache decorator with async functions."""
    from dspy.clients.cache import request_cache

    # Mock the dspy.cache attribute
    with patch("dspy.cache", cache):
        # Define a test function
        @request_cache()
        async def test_function(prompt, model):
            return f"Response for {prompt} with {model}"

        # First call should compute the result
        result1 = await test_function(prompt="Hello", model="openai/gpt-4o-mini")
        assert result1 == "Response for Hello with openai/gpt-4o-mini"

        # Second call with same arguments should use cache
        with patch.object(cache, "get") as mock_get:
            mock_get.return_value = "Cached response"
            result2 = await test_function(prompt="Hello", model="openai/gpt-4o-mini")
            assert result2 == "Cached response"
            mock_get.assert_called_once()

        # Call with different arguments should compute again
        result3 = await test_function(prompt="Different", model="openai/gpt-4o-mini")
        assert result3 == "Response for Different with openai/gpt-4o-mini"


def test_cache_consistency_with_lm_call_modifies_the_request(cache):
    """Test that the cache is consistent with the LM call that modifies the request."""
    from dspy.clients.cache import request_cache

    # Mock the dspy.cache attribute
    with patch("dspy.cache", cache):
        # Define a test function
        @request_cache()
        def test_function(**kwargs):
            del kwargs["field_to_delete"]
            return kwargs

        # First call should compute the result
        test_function(field_to_delete="delete", field_to_keep="keep")

        # The cache key should use the original request, not the modified one
        assert (
            cache.get(
                {
                    "field_to_keep": "keep",
                    "_fn_identifier": f"{test_function.__module__}.{test_function.__qualname__}",
                }
            )
            is None
        )
        assert (
            cache.get(
                {
                    "field_to_keep": "keep",
                    "field_to_delete": "delete",
                    "_fn_identifier": f"{test_function.__module__}.{test_function.__qualname__}",
                }
            )
            is not None
        )


def test_cache_fallback_on_restricted_environment():
    """Test that DSPy gracefully falls back to memory-only cache when disk cache fails."""
    old_env = os.environ.get("DSPY_CACHEDIR")
    try:
        # Set an invalid cache directory that can't be created
        os.environ["DSPY_CACHEDIR"] = "/dev/null/invalid_path"

        import dspy
        from dspy.clients import _get_dspy_cache

        dspy.cache = _get_dspy_cache()

        # Cache should work with memory-only fallback despite invalid disk path
        test_request = {"model": "test", "prompt": "hello"}
        dspy.cache.put(test_request, "fallback_result")
        result = dspy.cache.get(test_request)

        assert result == "fallback_result", "Memory cache fallback should work"

    finally:
        if old_env is None:
            os.environ.pop("DSPY_CACHEDIR", None)
        else:
            os.environ["DSPY_CACHEDIR"] = old_env


# ---------------------------------------------------------------------------
# Tests from test_cache_key_isolation.py
# ---------------------------------------------------------------------------


class TestCacheKeyIsolation:
    def test_fn_identifier_isolation(self, cache):
        """Different _fn_identifier values produce different cache keys."""
        base_request = {
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        request_sync = {**base_request, "_fn_identifier": "dspy.clients.lm.litellm_completion"}
        request_async = {**base_request, "_fn_identifier": "dspy.clients.lm.alitellm_completion"}

        key_sync = cache.cache_key(request_sync)
        key_async = cache.cache_key(request_async)

        assert key_sync != key_async

    def test_model_param_isolation(self, cache):
        """Different model values produce different keys."""
        base_request = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "messages": [{"role": "user", "content": "hello"}],
        }
        request_nano = {**base_request, "model": "openai/gpt-5-nano"}
        request_mini = {**base_request, "model": "openai/gpt-5-mini"}

        key_nano = cache.cache_key(request_nano)
        key_mini = cache.cache_key(request_mini)

        assert key_nano != key_mini

    def test_api_credentials_excluded(self, cache):
        """api_key, api_base, and base_url don't affect the cache key when excluded."""
        request_base = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        request_with_creds = {
            **request_base,
            "api_key": "test-api-key-placeholder",
            "api_base": "https://custom.openai.com/v1",
            "base_url": "https://custom.openai.com",
        }

        # The @request_cache decorator passes ignored_args_for_cache_key=["api_key", "api_base", "base_url"]
        ignored = ["api_key", "api_base", "base_url"]
        key_base = cache.cache_key(request_base, ignored_args_for_cache_key=ignored)
        key_with_creds = cache.cache_key(request_with_creds, ignored_args_for_cache_key=ignored)

        assert key_base == key_with_creds

        # Also verify that without the ignored list, credentials DO affect the key
        key_no_ignore = cache.cache_key(request_with_creds)
        assert key_base != key_no_ignore

    def test_rollout_id_produces_distinct_keys(self, cache):
        """Different rollout_id values produce different keys; same rollout_id produces same key."""
        base_request = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        request_rollout_1 = {**base_request, "rollout_id": 1}
        request_rollout_2 = {**base_request, "rollout_id": 2}
        request_rollout_1_again = {**base_request, "rollout_id": 1}

        key_1 = cache.cache_key(request_rollout_1)
        key_2 = cache.cache_key(request_rollout_2)
        key_1_again = cache.cache_key(request_rollout_1_again)

        assert key_1 != key_2
        assert key_1 == key_1_again

    def test_key_order_independence(self, cache):
        """Same key-value pairs in different insertion order produce the same cache key (OPT_SORT_KEYS)."""
        request_order_a = {
            "model": "openai/gpt-5-nano",
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
        }
        # Build the same dict with different insertion order
        request_order_b = {}
        request_order_b["temperature"] = 0.7
        request_order_b["messages"] = [{"role": "user", "content": "hello"}]
        request_order_b["model"] = "openai/gpt-5-nano"
        request_order_b["_fn_identifier"] = "dspy.clients.lm.litellm_completion"

        key_a = cache.cache_key(request_order_a)
        key_b = cache.cache_key(request_order_b)

        assert key_a == key_b

    def test_sync_async_fn_identifier_different(self, cache):
        """Sync litellm_completion vs async alitellm_completion identifiers produce different keys."""
        base_request = {
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        request_sync = {**base_request, "_fn_identifier": "dspy.clients.lm.litellm_completion"}
        request_async = {**base_request, "_fn_identifier": "dspy.clients.lm.alitellm_completion"}

        key_sync = cache.cache_key(request_sync)
        key_async = cache.cache_key(request_async)

        assert key_sync != key_async
        # Verify both are valid SHA-256 hashes
        assert len(key_sync) == 64
        assert len(key_async) == 64

    def test_rollout_id_none_equals_absent(self, cache):
        """rollout_id=None request produces same key as request without rollout_id.

        The LM class strips rollout_id when its value is None before the request
        reaches cache_key(). This test verifies that after that stripping, the keys
        match — i.e., a dict without rollout_id and a dict that had rollout_id=None
        removed both produce the same key.
        """
        request_without_rollout = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        # Simulate what LM.forward() does: pop rollout_id if it's None
        request_with_none_rollout = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
            "rollout_id": None,
        }
        # LM strips rollout_id=None before cache lookup
        if request_with_none_rollout.get("rollout_id") is None:
            request_with_none_rollout.pop("rollout_id", None)

        key_without = cache.cache_key(request_without_rollout)
        key_stripped = cache.cache_key(request_with_none_rollout)

        assert key_without == key_stripped

    def test_temperature_isolation(self, cache):
        """Different temperature values produce different cache keys."""
        base_request = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        request_temp_07 = {**base_request, "temperature": 0.7}
        request_temp_10 = {**base_request, "temperature": 1.0}

        key_07 = cache.cache_key(request_temp_07)
        key_10 = cache.cache_key(request_temp_10)

        assert key_07 != key_10


# ---------------------------------------------------------------------------
# Tests from test_cache_error_handling.py
# ---------------------------------------------------------------------------


class TestCorruptSQLiteDBFallback:
    """VAL-ERR-001: Corrupt SQLite DB file graceful degradation."""

    def test_corrupt_sqlite_db_fallback(self, tmp_path, monkeypatch):
        """When the SQLite DB file is corrupted, _get_dspy_cache() falls back to
        memory-only cache without crashing."""
        from dspy.clients import _get_dspy_cache

        cache_dir = str(tmp_path / "corrupt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, "dspy_cache.db")

        # Write garbage to the DB file so SQLite can't open it
        with open(db_path, "wb") as f:
            f.write(b"THIS IS NOT A SQLITE DATABASE" * 100)

        monkeypatch.setenv("DSPY_CACHEDIR", cache_dir)

        cache = _get_dspy_cache()

        # Cache should work with memory-only fallback
        assert cache.enable_memory_cache is True
        assert cache.enable_disk_cache is False

        # Verify the memory-only cache is actually functional
        request = {"model": "test", "prompt": "hello"}
        cache.put(request, "fallback_result")
        result = cache.get(request)
        assert result == "fallback_result"


class TestReadOnlyDatabasePut:
    """VAL-ERR-002: Read-only directory cache write behavior."""

    def test_readonly_database_put_doesnt_crash(self, tmp_path, caplog):
        """Cache.put() on a read-only DB logs a debug message but does not raise.

        We simulate a read-only DB by patching SQLiteCache.__setitem__ to raise
        sqlite3.OperationalError("attempt to write a readonly database"),
        which is the actual error SQLite raises when the database is read-only.
        Direct os.chmod on the DB file is unreliable across platforms due to WAL mode.
        """
        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )

        # Put a value first while the DB is writable
        request_readable = {"model": "test", "prompt": "readable"}
        cache.put(request_readable, "readable_value")

        # Enable log propagation so caplog can capture the debug message
        dspy_logger = logging.getLogger("dspy")
        original_propagate = dspy_logger.propagate
        dspy_logger.propagate = True

        try:
            request = {"model": "test", "prompt": "readonly_test"}
            with patch.object(
                type(cache.disk_cache),
                "__setitem__",
                side_effect=sqlite3.OperationalError("attempt to write a readonly database"),
            ):
                with caplog.at_level(logging.DEBUG, logger="dspy.clients.cache"):
                    # Should not raise
                    cache.put(request, "some_value")

            # Verify debug message was logged about the disk cache failure
            assert any("Failed to put value in disk cache" in rec.message for rec in caplog.records)

            # Memory cache should still have the value
            key = cache.cache_key(request)
            assert key in cache.memory_cache

        finally:
            dspy_logger.propagate = original_propagate

    def test_readonly_database_reads_still_work(self, tmp_path):
        """After making DB read-only, reads of previously-written values still work."""
        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )

        request = {"model": "test", "prompt": "before_readonly"}
        cache.put(request, "stored_value")

        # Clear memory so we must read from disk
        cache.reset_memory_cache()

        # Make the DB file read-only
        db_path = os.path.join(str(tmp_path), "dspy_cache.db")
        os.chmod(db_path, 0o444)

        try:
            result = cache.get(request)
            assert result == "stored_value"
        finally:
            os.chmod(db_path, 0o644)


class TestCorruptPickleLoadMemoryCache:
    """VAL-ERR-003: Corrupt pickle file for memory cache."""

    def test_corrupt_pickle_load_memory_cache(self, tmp_path):
        """Corrupt .pkl file passed to load_memory_cache raises
        cloudpickle.pickle.UnpicklingError (not an unhandled crash).

        NOTE: load_memory_cache() does NOT catch deserialization errors —
        corrupt pickle files raise UnpicklingError. This is documented as
        a discovered issue.
        """
        import pickle

        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=False,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=0,
            memory_max_entries=100,
        )

        # Write garbage bytes to a .pkl file
        pkl_path = str(tmp_path / "corrupt.pkl")
        with open(pkl_path, "wb") as f:
            f.write(b"\x80\x05\x95CORRUPT_GARBAGE_DATA_NOT_VALID_PICKLE")

        # load_memory_cache does not catch deserialization errors;
        # it raises UnpicklingError from cloudpickle/pickle
        with pytest.raises((pickle.UnpicklingError, Exception)):
            cache.load_memory_cache(pkl_path, allow_pickle=True)


class TestNonSerializableValueWarns:
    """VAL-ERR-004: Non-serializable value emits warning."""

    def test_non_serializable_tuple_warns(self, cache):
        """Cache.put() with a tuple emits UserWarning and doesn't crash."""
        request = {"model": "test", "prompt": "tuple_test"}
        with pytest.warns(UserWarning, match="Skipping disk cache write"):
            cache.put(request, (1, 2, 3))

        # Memory cache should still have the value
        result = cache.get(request)
        assert result == (1, 2, 3)

    def test_non_serializable_dataclass_warns(self, cache):
        """Cache.put() with a dataclass emits UserWarning and doesn't crash."""

        @dataclass
        class MyData:
            x: int
            y: str

        request = {"model": "test", "prompt": "dataclass_test"}
        with pytest.warns(UserWarning, match="Skipping disk cache write"):
            cache.put(request, MyData(x=42, y="hello"))

        # Memory cache should still have the value
        result = cache.get(request)
        assert result == MyData(x=42, y="hello")

    def test_non_serializable_doesnt_corrupt_other_entries(self, cache):
        """After a non-serializable write, other cache entries are unaffected."""
        good_request = {"model": "test", "prompt": "good_value"}
        cache.put(good_request, "good")

        bad_request = {"model": "test", "prompt": "bad_value"}
        with pytest.warns(UserWarning, match="Skipping disk cache write"):
            cache.put(bad_request, (1, 2, 3))

        # Original entry should still be accessible from disk
        cache.reset_memory_cache()
        assert cache.get(good_request) == "good"


class TestMemoryMissFallsToDisk:
    """VAL-ERR-005: Memory cache miss falls through to disk."""

    def test_memory_miss_falls_to_disk(self, cache):
        """After memory cache reset, Cache.get() returns the value from disk."""
        request = {"model": "test", "prompt": "disk_fallback"}
        cache.put(request, "persisted_value")

        # Verify value is in memory
        key = cache.cache_key(request)
        assert key in cache.memory_cache

        # Reset memory cache
        cache.reset_memory_cache()
        assert key not in cache.memory_cache

        # Get should fall through to disk and return the value
        result = cache.get(request)
        assert result == "persisted_value"

        # After the disk hit, value should be promoted back to memory
        assert key in cache.memory_cache


class TestSQLiteErrorInPutHandled:
    """VAL-ERR-006: sqlite3.Error in Cache.put() handled gracefully."""

    def test_sqlite_error_in_put_handled(self, tmp_path, caplog):
        """Mocked sqlite3.OperationalError in disk write is caught, memory write
        still succeeds."""
        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )

        request = {"model": "test", "prompt": "sqlite_error_test"}

        # Enable log propagation so caplog can capture the debug message
        dspy_logger = logging.getLogger("dspy")
        original_propagate = dspy_logger.propagate
        dspy_logger.propagate = True

        try:
            with patch.object(
                type(cache.disk_cache),
                "__setitem__",
                side_effect=sqlite3.OperationalError("disk I/O error"),
            ):
                with caplog.at_level(logging.DEBUG, logger="dspy.clients.cache"):
                    # Should not raise
                    cache.put(request, "survived_value")

            # Debug message should be logged
            assert any("Failed to put value in disk cache" in rec.message for rec in caplog.records)

            # Memory cache should have the value even though disk write failed
            result = cache.get(request)
            assert result == "survived_value"
        finally:
            dspy_logger.propagate = original_propagate

    def test_sqlite_error_memory_value_accessible(self, tmp_path):
        """When disk write raises sqlite3.Error, the value is still accessible
        from memory cache on subsequent get()."""
        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )

        request = {"model": "test", "prompt": "accessible_after_error"}

        with patch.object(
            type(cache.disk_cache),
            "__setitem__",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            cache.put(request, "memory_only_value")

        # Value should be retrievable from memory
        assert cache.get(request) == "memory_only_value"

        # But after resetting memory, it's gone (wasn't written to disk)
        cache.reset_memory_cache()
        assert cache.get(request) is None
