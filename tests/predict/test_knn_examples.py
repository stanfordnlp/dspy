import numpy as np

import dspy
from dspy.clients.embedding import Embedder
from dspy.predict.knn import KNN


def _embedder():
    def encode(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    return Embedder(encode)


def test_knn_builds_without_with_inputs():
    """The class docstring uses examples that never call with_inputs()."""
    trainset = [
        dspy.Example(input="hello", output="world"),
        dspy.Example(input="foo", output="bar"),
    ]
    knn = KNN(k=1, trainset=trainset, vectorizer=_embedder())
    neighbors = knn(input="hello")
    assert len(neighbors) == 1
    assert neighbors[0].output in {"world", "bar"}
