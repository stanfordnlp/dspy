import pytest
import numpy as np
from dspy.embeddings.metrics import (
    SimilarityMetric,
    cosine_similarity,
    euclidean_similarity,
    manhattan_similarity
)


def test_similarity_metric_enum():
    assert SimilarityMetric.COSINE == 'cosine'
    assert SimilarityMetric.EUCLIDEAN == 'euclidean'
    assert SimilarityMetric.MANHATTAN == 'manhattan'


def test_cosine_similarity():
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    similarity = cosine_similarity(v1, v2)
    assert similarity == pytest.approx(1.0)

    v2 = np.array([0, 1, 0])
    similarity = cosine_similarity(v1, v2)
    assert similarity == pytest.approx(0.0)

    v2 = np.array([-1, 0, 0])
    similarity = cosine_similarity(v1, v2)
    assert similarity == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="Zero vector encountered"):
        cosine_similarity(v1, v2)


# def test_euclidean_similarity():
#     v1 = np.array([1, 0, 0])
#     v2 = np.array([1, 0, 0])
#     similarity = euclidean_similarity(v1, v2)
#     assert similarity == pytest.approx(1.0)

#     v2 = np.array([0, 0, 0])
#     similarity = euclidean_similarity(v1, v2)
#     assert similarity == pytest.approx(0.0)


def test_manhattan_similarity():
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    similarity = manhattan_similarity(v1, v2)
    assert similarity == pytest.approx(1.0)

    v2 = np.array([0, 0, 0])
    similarity = manhattan_similarity(v1, v2)
    assert similarity == pytest.approx(0.6666666666)


def test_similarity_functions_clip():
    v1 = np.array([1e100, 1e100, 1e100])
    v2 = np.array([-1e100, -1e100, -1e100])
    similarity = cosine_similarity(v1, v2)
    assert similarity == pytest.approx(-1.0)


def test_similarity_functions_different_lengths():
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0])
    with pytest.raises(ValueError):
        cosine_similarity(v1, v2)
    with pytest.raises(ValueError):
        euclidean_similarity(v1, v2)
    with pytest.raises(ValueError):
        manhattan_similarity(v1, v2)
