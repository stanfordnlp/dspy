from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import dspy
from dspy.clients.embedding import Embedder


# Mock response format similar to litellm's embedding response.
class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [{"embedding": emb} for emb in embeddings]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model = "mock_model"
        self.object = "list"


@pytest.fixture
def cache(tmp_path):
    original_cache = dspy.cache
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache")
    yield
    dspy.cache = original_cache


def test_litellm_embedding(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello", "world"]
    mock_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
    ]

    with patch("litellm.embedding") as mock_litellm:
        # Configure mock to return proper response format.
        mock_litellm.return_value = MockEmbeddingResponse(mock_embeddings)

        # Create embedding instance and call it.
        embedding = Embedder(model, caching=True)
        result = embedding(inputs)

        # Verify litellm was called with correct parameters.
        # Because we disable the litellm cache, it should be called with caching=False.
        mock_litellm.assert_called_once_with(model=model, input=inputs, caching=False)

        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)

        # Second call should be cached.
        result = embedding(inputs)
        assert mock_litellm.call_count == 1
        np.testing.assert_allclose(result, mock_embeddings)

        # Disable cache should issue new calls.
        embedding = Embedder(model, caching=False)
        result = embedding(inputs)
        assert mock_litellm.call_count == 2
        np.testing.assert_allclose(result, mock_embeddings)


@pytest.mark.parametrize("endpoint_key", ["api_base", "base_url"])
def test_embedding_cache_uses_endpoint_but_ignores_api_key(cache, endpoint_key):
    endpoint_a = "https://endpoint-a.example/v1"
    endpoint_b = "https://endpoint-b.example/v1"
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs.copy())
        value = 1.0 if kwargs[endpoint_key] == endpoint_a else 2.0
        return MockEmbeddingResponse([[value]])

    with patch("litellm.embedding", side_effect=fake_embedding):
        embedding = Embedder("text-embedding-ada-002", caching=True)
        first = embedding(["hello"], **{endpoint_key: endpoint_a, "api_key": "key-a"})
        cached = embedding(["hello"], **{endpoint_key: endpoint_a, "api_key": "key-b"})
        second_endpoint = embedding(["hello"], **{endpoint_key: endpoint_b, "api_key": "key-b"})

    np.testing.assert_allclose(first, [[1.0]])
    np.testing.assert_allclose(cached, [[1.0]])
    np.testing.assert_allclose(second_endpoint, [[2.0]])
    assert [(call[endpoint_key], call["api_key"]) for call in calls] == [
        (endpoint_a, "key-a"),
        (endpoint_b, "key-b"),
    ]


def test_callable_embedding(cache):
    inputs = ["hello", "world", "test"]

    expected_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
        [0.7, 0.8, 0.9],  # embedding for "test"
    ]

    class EmbeddingFn:
        def __init__(self):
            self.call_count = 0

        def __call__(self, texts):
            # Simple callable that returns random embeddings.
            self.call_count += 1
            return expected_embeddings

    embedding_fn = EmbeddingFn()

    # Create embedding instance with callable
    embedding = Embedder(embedding_fn)
    result = embedding(inputs)

    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)

    result = embedding(inputs)
    # The second call should be cached.
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)


def test_callable_numpy_embedding_persists_to_disk(cache, tmp_path):
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache_safe", restrict_pickle=True)

    inputs = ["hello", "world"]
    expected_embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    embedding_fn = MagicMock(return_value=expected_embeddings)
    embedding = Embedder(embedding_fn)

    result = embedding(inputs)
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)

    result = embedding(inputs)
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)

    dspy.cache.reset_memory_cache()

    result = embedding(inputs)
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)


def test_invalid_model_type():
    # Test that invalid model type raises ValueError
    with pytest.raises(ValueError):
        embedding = Embedder(123)  # Invalid model type
        embedding(["test"])


@pytest.mark.asyncio
async def test_async_embedding():
    model = "text-embedding-ada-002"
    inputs = ["hello", "world"]
    mock_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
    ]

    with patch("litellm.aembedding") as mock_litellm:
        # Configure mock to return proper response format.
        mock_litellm.return_value = MockEmbeddingResponse(mock_embeddings)

        # Create embedding instance and call it.
        embedding = Embedder(model, caching=False)
        result = await embedding.acall(inputs)

        # Verify litellm was called with correct parameters.
        mock_litellm.assert_called_once_with(model=model, input=inputs, caching=False)

        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_key", ["api_base", "base_url"])
async def test_async_embedding_cache_uses_endpoint_but_ignores_api_key(cache, endpoint_key):
    endpoint_a = "https://endpoint-a.example/v1"
    endpoint_b = "https://endpoint-b.example/v1"
    calls = []

    async def fake_embedding(**kwargs):
        calls.append(kwargs.copy())
        value = 1.0 if kwargs[endpoint_key] == endpoint_a else 2.0
        return MockEmbeddingResponse([[value]])

    with patch("litellm.aembedding", side_effect=fake_embedding):
        embedding = Embedder("text-embedding-ada-002", caching=True)
        first = await embedding.acall(["hello"], **{endpoint_key: endpoint_a, "api_key": "key-a"})
        cached = await embedding.acall(["hello"], **{endpoint_key: endpoint_a, "api_key": "key-b"})
        second_endpoint = await embedding.acall(["hello"], **{endpoint_key: endpoint_b, "api_key": "key-b"})

    np.testing.assert_allclose(first, [[1.0]])
    np.testing.assert_allclose(cached, [[1.0]])
    np.testing.assert_allclose(second_endpoint, [[2.0]])
    assert [(call[endpoint_key], call["api_key"]) for call in calls] == [
        (endpoint_a, "key-a"),
        (endpoint_b, "key-b"),
    ]


def test_call_caching_false_overrides_instance_true(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.embedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=True)
        embedding(inputs)
        embedding(inputs, caching=False)
        assert mock_litellm.call_count == 2


def test_call_caching_true_overrides_instance_false(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.embedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=False)
        embedding(inputs, caching=True)
        embedding(inputs, caching=True)
        assert mock_litellm.call_count == 1


@pytest.mark.asyncio
async def test_acall_caching_false_overrides_instance_true(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.aembedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=True)
        await embedding.acall(inputs)
        await embedding.acall(inputs, caching=False)
        assert mock_litellm.call_count == 2


@pytest.mark.asyncio
async def test_acall_caching_true_overrides_instance_false(cache):
    model = "text-embedding-ada-002"
    inputs = ["hello"]
    with patch("litellm.aembedding") as mock_litellm:
        mock_litellm.return_value = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
        embedding = Embedder(model, caching=False)
        await embedding.acall(inputs, caching=True)
        await embedding.acall(inputs, caching=True)
        assert mock_litellm.call_count == 1
