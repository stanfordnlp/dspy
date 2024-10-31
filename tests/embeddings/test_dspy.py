import pytest
from dspy import DSPy, dspy
from dspy.embeddings.providers.lite_embedder import LiteEmbedder
from dspy.embeddings.config import EmbeddingConfig


def test_dspy_LiteEmbedder():
    model = 'test-model'
    api_key = 'test-key'
    embedder = DSPy.LiteEmbedder(model=model, api_key=api_key)
    assert isinstance(embedder, LiteEmbedder)
    assert embedder.model == model


def test_dspy_instance():
    from dspy import dspy
    assert isinstance(dspy, DSPy)
