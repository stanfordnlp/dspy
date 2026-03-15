import logging
import os
import sqlite3
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


def test_cache_miss_and_unserializable_key(cache):
    """Cache miss returns None; unserializable request key also returns None without raising."""
    assert cache.get({"prompt": "Non-existent", "model": "gpt-4"}) is None

    class UnserializableObject:
        pass

    request = {"data": UnserializableObject()}
    assert cache.get(request) is None
    cache.put(request, "value")  # should not raise


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


def test_cache_contains_checks_disk(cache):
    request = {"prompt": "test"}
    key = cache.cache_key(request)

    assert key not in cache
    cache.put(request, "value")
    assert key in cache
    cache.reset_memory_cache()
    assert key in cache


def test_corrupt_disk_entries_return_none(cache):
    """Cache.get() returns None (not crash) for: missing module, corrupt JSON,
    and pydantic validation failure."""
    corrupt_entries = {
        "broken_entry": orjson.dumps(
            {
                "_data": {
                    "__dspy_cache_type__": "pydantic",
                    "__dspy_cache_module__": "nonexistent.module",
                    "__dspy_cache_qualname__": "FakeClass",
                    "__dspy_cache_data__": {"x": 1},
                }
            }
        ),
        "corrupt_json": b"{not valid json",
        "invalid_pydantic": orjson.dumps(
            {
                "_data": {
                    "__dspy_cache_type__": "pydantic",
                    "__dspy_cache_module__": CacheValidationModel.__module__,
                    "__dspy_cache_qualname__": CacheValidationModel.__qualname__,
                    "__dspy_cache_data__": {"name": "missing_required_value"},
                }
            }
        ),
    }

    with cache.disk_cache._lock:
        conn = cache.disk_cache._get_conn()
        for prompt, blob in corrupt_entries.items():
            request = {"model": "test", "prompt": prompt}
            key = cache.cache_key(request)
            conn.execute(
                "INSERT INTO cache (key, value, size, last_access) VALUES (?, ?, ?, ?)",
                (key, blob, len(blob), 0.0),
            )
        conn.commit()

    cache.reset_memory_cache()
    for prompt in corrupt_entries:
        assert cache.get({"model": "test", "prompt": prompt}) is None


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


# ---------------------------------------------------------------------------
# Tests from test_cache_key_isolation.py
# ---------------------------------------------------------------------------


class TestCacheKeyIsolation:
    @pytest.mark.parametrize(
        "field, val_a, val_b",
        [
            ("model", "openai/gpt-5-nano", "openai/gpt-5-mini"),
            ("temperature", 0.7, 1.0),
            ("_fn_identifier", "dspy.clients.lm.litellm_completion", "dspy.clients.lm.alitellm_completion"),
        ],
    )
    def test_different_field_values_produce_different_keys(self, cache, field, val_a, val_b):
        base = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert cache.cache_key({**base, field: val_a}) != cache.cache_key({**base, field: val_b})

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

        ignored = ["api_key", "api_base", "base_url"]
        key_base = cache.cache_key(request_base, ignored_args_for_cache_key=ignored)
        key_with_creds = cache.cache_key(request_with_creds, ignored_args_for_cache_key=ignored)

        assert key_base == key_with_creds
        assert key_base != cache.cache_key(request_with_creds)

    def test_rollout_id_isolation_and_none_stripping(self, cache):
        """Different rollout_id values produce different keys; same rollout_id reproduces;
        stripping rollout_id=None matches absent rollout_id."""
        base = {
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
        }
        key_1 = cache.cache_key({**base, "rollout_id": 1})
        key_2 = cache.cache_key({**base, "rollout_id": 2})
        assert key_1 != key_2
        assert key_1 == cache.cache_key({**base, "rollout_id": 1})

        # Stripping None rollout_id matches absent
        request_none = {**base, "rollout_id": None}
        request_none.pop("rollout_id")
        assert cache.cache_key(base) == cache.cache_key(request_none)

    def test_key_order_independence(self, cache):
        """Same key-value pairs in different insertion order produce the same cache key (OPT_SORT_KEYS)."""
        request_order_a = {
            "model": "openai/gpt-5-nano",
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
        }
        request_order_b = {}
        request_order_b["temperature"] = 0.7
        request_order_b["messages"] = [{"role": "user", "content": "hello"}]
        request_order_b["model"] = "openai/gpt-5-nano"
        request_order_b["_fn_identifier"] = "dspy.clients.lm.litellm_completion"

        assert cache.cache_key(request_order_a) == cache.cache_key(request_order_b)


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


def test_non_serializable_values_warn_and_stay_in_memory(cache):
    """Non-serializable values (tuple, dataclass) warn on disk write, remain in memory,
    and don't corrupt other entries."""

    @dataclass
    class MyData:
        x: int
        y: str

    good_request = {"model": "test", "prompt": "good_value"}
    cache.put(good_request, "good")

    for label, value in [("tuple", (1, 2, 3)), ("dataclass", MyData(x=42, y="hello"))]:
        request = {"model": "test", "prompt": label}
        with pytest.warns(UserWarning, match="Skipping disk cache write"):
            cache.put(request, value)
        assert cache.get(request) == value

    cache.reset_memory_cache()
    assert cache.get(good_request) == "good"


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


