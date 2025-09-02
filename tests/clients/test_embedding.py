from unittest.mock import patch

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
