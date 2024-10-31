import pytest
import numpy as np
import torch
from dspy.embeddings.base import Embedder, EmbeddingError
from dspy.embeddings.config import EmbeddingConfig, OutputFormat
from dspy.embeddings.metrics import SimilarityMetric


class DummyEmbedder(Embedder):
    def embed(self, texts, output_format=None):
        # Return dummy embeddings
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        embeddings = [[1.0, 2.0, 3.0] for _ in texts]
        output_format = output_format or self.config.default_output_format
        embeddings = self._convert_output_format(embeddings, output_format)
        return embeddings[0] if single else embeddings


def test_embedder_convert_output_format():
    embedder = DummyEmbedder()
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    # Test list output
    output = embedder._convert_output_format(embeddings, OutputFormat.LIST)
    assert isinstance(output, list)
    assert output == embeddings

    # Test array output
    output = embedder._convert_output_format(embeddings, OutputFormat.ARRAY)
    assert isinstance(output, np.ndarray)
    assert output.shape == (2, 3)

    # Test tensor output
    output = embedder._convert_output_format(embeddings, OutputFormat.TENSOR)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3)


def test_embedder_similarity():
    embedder = DummyEmbedder()
    text1 = "hello"
    text2 = "world"
    similarity = embedder.similarity(text1, text2)
    assert isinstance(similarity, float)


def test_embedder_similarity_metrics():
    embedder = DummyEmbedder()
    text1 = "hello"
    text2 = "world"

    cosine_sim = embedder.similarity(text1, text2, metric=SimilarityMetric.COSINE)
    euclidean_sim = embedder.similarity(text1, text2, metric=SimilarityMetric.EUCLIDEAN)
    manhattan_sim = embedder.similarity(text1, text2, metric=SimilarityMetric.MANHATTAN)

    assert isinstance(cosine_sim, float)
    assert isinstance(euclidean_sim, float)
    assert isinstance(manhattan_sim, float)


def test_embedder_cannot_instantiate():
    with pytest.raises(TypeError):
        embedder = Embedder()


def test_embedder_invalid_metric():
    embedder = DummyEmbedder()
    text1 = "hello"
    text2 = "world"
    with pytest.raises(EmbeddingError, match="Similarity calculation error"):
        embedder.similarity(text1, text2, metric='invalid_metric')
