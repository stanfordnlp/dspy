import os
from dataclasses import dataclass
from unittest.mock import patch

import pydantic
import pytest
from cachetools import LRUCache
from diskcache import FanoutCache

from dspy.clients.cache import Cache


@dataclass
class DummyResponse:
    message: str
    usage: dict


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
    assert isinstance(disk_cache.disk_cache, FanoutCache)
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

    value = DummyResponse(message="This is a test response", usage={"prompt_tokens": 10, "completion_tokens": 20})

    cache.put(request, value)
    result = cache.get(request)

    assert result.message == value.message
    assert result.usage == {}

    # Test with disk cache
    # First, clear memory cache to ensure we're using disk cache
    cache.reset_memory_cache()

    # Get from disk cache
    result_from_disk = cache.get(request)
    assert result_from_disk.message == value.message
    assert result_from_disk.usage == {}

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
