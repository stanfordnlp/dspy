import pytest
from unittest.mock import patch
import numpy as np
import torch
from dspy.embeddings.providers.lite_embedder import LiteEmbedder, EmbeddingError
from dspy.embeddings.config import EmbeddingConfig, OutputFormat
from dspy.embeddings.metrics import SimilarityMetric


def mock_embedding(*args, **kwargs):
    # Mocked response similar to litellm.embedding
    inputs = kwargs.get('input')
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [{'embedding': [0.1, 0.2, 0.3]} for _ in inputs]
    return {'data': data}


@patch('dspy.embeddings.providers.lite_embedder.embedding', side_effect=mock_embedding)
def test_lite_embedder_embed_single_text(mock_embedding_func):
    embedder = LiteEmbedder(model='test-model', api_key='test-key')
    text = "hello world"
    embedding = embedder.embed(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 3


@patch('dspy.embeddings.providers.lite_embedder.embedding', side_effect=mock_embedding)
def test_lite_embedder_embed_multiple_texts(mock_embedding_func):
    embedder = LiteEmbedder(model='test-model', api_key='test-key')
    texts = ["hello", "world"]
    embeddings = embedder.embed(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(len(e) == 3 for e in embeddings)


@patch('dspy.embeddings.providers.lite_embedder.embedding', side_effect=mock_embedding)
def test_lite_embedder_output_format_array(mock_embedding_func):
    embedder = LiteEmbedder(
        model='test-model',
        api_key='test-key',
        config=EmbeddingConfig(default_output_format=OutputFormat.ARRAY)
    )
    texts = ["hello", "world"]
    embeddings = embedder.embed(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 3)


@patch('dspy.embeddings.providers.lite_embedder.embedding', side_effect=mock_embedding)
def test_lite_embedder_output_format_tensor(mock_embedding_func):
    embedder = LiteEmbedder(model='test-model', api_key='test-key')
    texts = ["hello", "world"]
    embeddings = embedder.embed(texts, output_format=OutputFormat.TENSOR)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 3)


@patch('dspy.embeddings.providers.lite_embedder.embedding', side_effect=mock_embedding)
def test_lite_embedder_similarity(mock_embedding_func):
    embedder = LiteEmbedder(model='test-model', api_key='test-key')
    text1 = "hello"
    text2 = "world"
    similarity = embedder.similarity(text1, text2)
    assert isinstance(similarity, float)


@patch('dspy.embeddings.providers.lite_embedder.embedding', side_effect=mock_embedding)
def test_lite_embedder_caching(mock_embedding_func):
    config = EmbeddingConfig(cache_embeddings=True)
    embedder = LiteEmbedder(model='test-model', api_key='test-key', config=config)
    texts = ["hello", "world"]
    embeddings1 = embedder.embed(texts)
    embeddings2 = embedder.embed(texts)
    assert embeddings1 == embeddings2
    # Should have been called only once due to caching
    assert mock_embedding_func.call_count == 1


@patch('dspy.embeddings.providers.lite_embedder.embedding', side_effect=Exception("API error"))
def test_lite_embedder_api_error(mock_embedding_func):
    embedder = LiteEmbedder(model='test-model', api_key='test-key')
    texts = ["hello"]
    with pytest.raises(EmbeddingError, match="API error"):
        embeddings = embedder.embed(texts)


def test_lite_embedder_missing_model_or_api_key():
    with pytest.raises(ValueError, match="Model and API key must be provided"):
        embedder = LiteEmbedder(model=None, api_key='test-key')
    with pytest.raises(ValueError, match="Model and API key must be provided"):
        embedder = LiteEmbedder(model='test-model', api_key=None)



