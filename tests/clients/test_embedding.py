import pytest
from unittest.mock import Mock, patch
import numpy as np

from dspy.clients.embedding import Embedding


# Mock response format similar to litellm's embedding response.
class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [{"embedding": emb} for emb in embeddings]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model = "mock_model"
        self.object = "list"


def test_litellm_embedding():
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
        embedding = Embedding(model)
        result = embedding(inputs)

        # Verify litellm was called with correct parameters.
        mock_litellm.assert_called_once_with(model=model, input=inputs, caching=True)

        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)


def test_callable_embedding():
    inputs = ["hello", "world", "test"]

    expected_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
        [0.7, 0.8, 0.9],  # embedding for "test"
    ]

    def mock_embedding_fn(texts):
        # Simple callable that returns random embeddings.
        return expected_embeddings

    # Create embedding instance with callable
    embedding = Embedding(mock_embedding_fn)
    result = embedding(inputs)

    np.testing.assert_allclose(result, expected_embeddings)


def test_invalid_model_type():
    # Test that invalid model type raises ValueError
    with pytest.raises(ValueError):
        embedding = Embedding(123)  # Invalid model type
        embedding(["test"])
