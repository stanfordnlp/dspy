import numpy as np

import dspy
from dspy.clients.embedding import Embedder
from dspy.predict.knn import KNN


def test_knn_builds_without_with_inputs():
    """The class docstring uses examples that never call with_inputs()."""
    captured = []

    def encode(texts):
        captured.extend(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    trainset = [
        dspy.Example(input="hello", output="world"),
        dspy.Example(input="foo", output="bar"),
    ]
    knn = KNN(k=1, trainset=trainset, vectorizer=Embedder(encode))
    assert captured == [
        "input: hello | output: world",
        "input: foo | output: bar",
    ]
    neighbors = knn(input="hello")
    assert len(neighbors) == 1
    assert neighbors[0].output in {"world", "bar"}
    assert captured[-1] == "input: hello"
