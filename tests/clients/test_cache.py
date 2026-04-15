import os
from dataclasses import dataclass
from unittest.mock import patch

import pydantic
import pytest
from cachetools import LRUCache
from diskcache import FanoutCache

import dspy
from dspy.clients.cache import Cache
from dspy.clients.disk_serialization import DeserializationError


@dataclass
class DummyResponse:
    message: str
    usage: dict


@dataclass
class CacheValidationDataclass:
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


@pytest.fixture
def orjson_cache(tmp_path):
    """Create a cache instance with the safe serializer backend."""
    return Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path / "orjson"),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        use_pickle=False,
    )


def test_initialization(tmp_path):
    """Test different cache initialization configurations."""
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

    disk_cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path / "pickle"),
        disk_size_limit_bytes=1024,
        memory_max_entries=0,
    )
    assert isinstance(disk_cache.disk_cache, FanoutCache)
    assert disk_cache.memory_cache == {}

    safe_cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path / "orjson"),
        disk_size_limit_bytes=1024,
        memory_max_entries=0,
        use_pickle=False,
    )
    assert isinstance(safe_cache.disk_cache, FanoutCache)
    assert safe_cache.use_pickle is False
    assert safe_cache.memory_cache == {}

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
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini", "temperature": 0.7}
    key = cache.cache_key(request)
    assert isinstance(key, str)
    assert len(key) == 64

    class TestModel(pydantic.BaseModel):
        name: str
        value: int

    model = TestModel(name="test", value=42)
    request_with_model = {"data": model}
    key_with_model = cache.cache_key(request_with_model)
    assert isinstance(key_with_model, str)

    request_with_model_class = {"model_class": TestModel}
    key_with_model_class = cache.cache_key(request_with_model_class)
    assert isinstance(key_with_model_class, str)

    @dataclass
    class TestDataclass:
        name: str
        value: int

    key_with_dataclass = cache.cache_key({"data": TestDataclass(name="test", value=42)})
    assert isinstance(key_with_dataclass, str)
    assert key_with_dataclass == cache.cache_key({"data": TestDataclass(name="test", value=42)})
    assert key_with_dataclass != cache.cache_key({"data": TestDataclass(name="test", value=43)})


def test_put_and_get(cache):
    """Test putting and getting from cache."""
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini", "temperature": 0.7}
    value = DummyResponse(message="This is a test response", usage={"prompt_tokens": 10, "completion_tokens": 20})

    cache.put(request, value)
    result = cache.get(request)

    assert result.message == value.message
    assert result.usage == {}

    cache.reset_memory_cache()

    result_from_disk = cache.get(request)
    assert result_from_disk.message == value.message
    assert result_from_disk.usage == {}
    assert cache.cache_key(request) in cache.memory_cache


def test_cache_miss_and_unserializable_key(cache):
    assert cache.get({"prompt": "Non-existent", "model": "gpt-4"}) is None

    class UnserializableObject:
        pass

    request = {"data": UnserializableObject()}
    assert cache.get(request) is None
    cache.put(request, "value")


def test_model_response_roundtrip_in_safe_mode(orjson_cache):
    from litellm import ModelResponse

    response = ModelResponse(
        id="test-123",
        choices=[{"message": {"content": "cached response"}, "index": 0, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    request = {"model": "openai/gpt-5-nano", "prompt": "test"}

    orjson_cache.put(request, response)
    orjson_cache.reset_memory_cache()

    result = orjson_cache.get(request)
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "cached response"


def test_registered_dataclass_roundtrip_in_safe_mode(tmp_path):
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        use_pickle=False,
        safe_types=[CacheValidationDataclass],
    )
    request = {
        "model": "test",
        "prompt": "registered_dataclass_test",
        "payload": CacheValidationDataclass(name="request", value=1),
    }
    response = CacheValidationDataclass(name="hello", value=3)

    cache.put(request, response)
    result = cache.get(request)

    assert isinstance(result, CacheValidationDataclass)
    assert result == response


def test_configure_cache_registers_safe_types(tmp_path):
    original_cache = dspy.cache
    try:
        dspy.configure_cache(
            enable_disk_cache=True,
            enable_memory_cache=False,
            disk_cache_dir=str(tmp_path / "configured"),
            use_pickle=False,
            safe_types=[CacheValidationDataclass],
        )
        request = {"model": "test", "prompt": "configured_safe_type"}
        response = CacheValidationDataclass(name="configured", value=7)

        dspy.cache.put(request, response)
        result = dspy.cache.get(request)

        assert isinstance(result, CacheValidationDataclass)
        assert result == response
    finally:
        dspy.cache = original_cache


def test_corrupt_disk_entries_return_none(orjson_cache):
    request = {"model": "test", "prompt": "will_be_corrupted"}
    orjson_cache.put(request, "good_value")
    orjson_cache.reset_memory_cache()

    with patch.object(type(orjson_cache.disk_cache), "get", side_effect=DeserializationError("corrupt")):
        assert orjson_cache.get(request) is None


def test_non_serializable_values_warn_and_stay_in_memory(orjson_cache):
    good_request = {"model": "test", "prompt": "good_value"}
    orjson_cache.put(good_request, "good")

    @dataclass
    class UnregisteredDataclass:
        value: int

    for label, value in [("tuple", (1, 2, 3)), ("dataclass", UnregisteredDataclass(value=1))]:
        request = {"model": "test", "prompt": label}
        with pytest.warns(UserWarning, match="Skipping disk cache write"):
            orjson_cache.put(request, value)
        assert orjson_cache.get(request) == value

    orjson_cache.reset_memory_cache()
    assert orjson_cache.get(good_request) == "good"


def test_reset_memory_cache(cache):
    """Test resetting memory cache."""
    requests = [{"prompt": f"Hello {i}", "model": "openai/gpt-4o-mini"} for i in range(5)]
    for i, req in enumerate(requests):
        cache.put(req, f"Response {i}")

    for req in requests:
        key = cache.cache_key(req)
        assert key in cache.memory_cache

    cache.reset_memory_cache()
    assert len(cache.memory_cache) == 0

    for req in requests:
        result = cache.get(req)
        assert result is not None


def test_save_and_load_memory_cache(cache, tmp_path):
    """Test saving and loading memory cache."""
    requests = [{"prompt": f"Hello {i}", "model": "openai/gpt-4o-mini"} for i in range(5)]
    for i, req in enumerate(requests):
        cache.put(req, f"Response {i}")

    temp_cache_file = tmp_path / "memory_cache.pkl"
    cache.save_memory_cache(str(temp_cache_file))

    new_cache = Cache(
        enable_memory_cache=True,
        enable_disk_cache=False,
        disk_cache_dir=tmp_path / "disk_cache",
        disk_size_limit_bytes=0,
        memory_max_entries=100,
    )

    with pytest.raises(ValueError):
        new_cache.load_memory_cache(str(temp_cache_file))

    new_cache.load_memory_cache(str(temp_cache_file), allow_pickle=True)

    for req in requests:
        result = new_cache.get(req)
        assert result is not None
        assert result == f"Response {requests.index(req)}"


def test_request_cache_decorator(cache):
    """Test the lm_cache decorator."""
    from dspy.clients.cache import request_cache

    with patch("dspy.cache", cache):
        @request_cache()
        def test_function(prompt, model):
            return f"Response for {prompt} with {model}"

        result1 = test_function(prompt="Hello", model="openai/gpt-4o-mini")
        assert result1 == "Response for Hello with openai/gpt-4o-mini"

        with patch.object(cache, "get") as mock_get:
            mock_get.return_value = "Cached response"
            result2 = test_function(prompt="Hello", model="openai/gpt-4o-mini")
            assert result2 == "Cached response"
            mock_get.assert_called_once()

        result3 = test_function(prompt="Different", model="openai/gpt-4o-mini")
        assert result3 == "Response for Different with openai/gpt-4o-mini"


def test_request_cache_decorator_with_ignored_args_for_cache_key(cache):
    """Test the request_cache decorator with ignored_args_for_cache_key."""
    from dspy.clients.cache import request_cache

    with patch("dspy.cache", cache):
        @request_cache(ignored_args_for_cache_key=["model"])
        def test_function1(prompt, model):
            return f"Response for {prompt} with {model}"

        @request_cache()
        def test_function2(prompt, model):
            return f"Response for {prompt} with {model}"

        result1 = test_function1(prompt="Hello", model="openai/gpt-4o-mini")
        result2 = test_function1(prompt="Hello", model="openai/gpt-4o")
        assert result1 == result2

        result3 = test_function2(prompt="Hello", model="openai/gpt-4o-mini")
        result4 = test_function2(prompt="Hello", model="openai/gpt-4o")
        assert result3 != result4


@pytest.mark.asyncio
async def test_request_cache_decorator_async(cache):
    """Test the request_cache decorator with async functions."""
    from dspy.clients.cache import request_cache

    with patch("dspy.cache", cache):
        @request_cache()
        async def test_function(prompt, model):
            return f"Response for {prompt} with {model}"

        result1 = await test_function(prompt="Hello", model="openai/gpt-4o-mini")
        assert result1 == "Response for Hello with openai/gpt-4o-mini"

        with patch.object(cache, "get") as mock_get:
            mock_get.return_value = "Cached response"
            result2 = await test_function(prompt="Hello", model="openai/gpt-4o-mini")
            assert result2 == "Cached response"
            mock_get.assert_called_once()

        result3 = await test_function(prompt="Different", model="openai/gpt-4o-mini")
        assert result3 == "Response for Different with openai/gpt-4o-mini"


def test_cache_consistency_with_lm_call_modifies_the_request(cache):
    """Test that the cache is consistent with the LM call that modifies the request."""
    from dspy.clients.cache import request_cache

    with patch("dspy.cache", cache):
        @request_cache()
        def test_function(**kwargs):
            del kwargs["field_to_delete"]
            return kwargs

        test_function(field_to_delete="delete", field_to_keep="keep")

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
        os.environ["DSPY_CACHEDIR"] = "/dev/null/invalid_path"

        import dspy
        from dspy.clients import _get_dspy_cache

        dspy.cache = _get_dspy_cache()

        test_request = {"model": "test", "prompt": "hello"}
        dspy.cache.put(test_request, "fallback_result")
        result = dspy.cache.get(test_request)

        assert result == "fallback_result"

    finally:
        if old_env is None:
            os.environ.pop("DSPY_CACHEDIR", None)
        else:
            os.environ["DSPY_CACHEDIR"] = old_env
