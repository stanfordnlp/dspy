import pytest

import dspy
from dspy.predict import KNN
from dspy.utils import DummyVectorizer

pytestmark = pytest.mark.extra


def mock_example(question: str, answer: str) -> dspy.Example:
    """Creates a mock DSP example with specified question and answer."""
    return dspy.Example(question=question, answer=answer).with_inputs("question")


@pytest.fixture
def setup_knn() -> KNN:
    """Sets up a KNN instance with a mocked vectorizer for testing."""
    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the largest ocean?", "Pacific"),
        mock_example("What is 2+2?", "4"),
    ]
    return KNN(k=2, trainset=trainset, vectorizer=dspy.Embedder(DummyVectorizer()))


def test_knn_initialization(setup_knn):
    """Tests the KNN initialization and checks if the trainset vectors are correctly created."""
    import numpy as np

    knn = setup_knn
    assert knn.k == 2, "Incorrect k value"
    assert len(knn.trainset_vectors) == 3, "Incorrect size of trainset vectors"
    assert isinstance(knn.trainset_vectors, np.ndarray), "Trainset vectors should be a NumPy array"


def test_knn_query(setup_knn):
    """Tests the KNN query functionality for retrieving the nearest neighbors."""
    knn = setup_knn
    query = {"question": "What is 3+3?"}  # A query close to "What is 2+2?"
    nearest_samples = knn(**query)
    assert len(nearest_samples) == 2, "Incorrect number of nearest samples returned"
    assert nearest_samples[0].answer == "4", "Incorrect nearest sample returned"


def test_knn_query_specificity(setup_knn):
    """Tests the KNN query functionality for specificity of returned examples."""
    knn = setup_knn
    query = {"question": "What is the capital of Germany?"}  # A query close to "What is the capital of France?"
    nearest_samples = knn(**query)
    assert len(nearest_samples) == 2, "Incorrect number of nearest samples returned"
    assert "Paris" in [sample.answer for sample in nearest_samples], "Expected Paris to be a nearest sample answer"


def test_knn_query_zero_k():
    """Tests that KNN with k=0 returns an empty list instead of the full trainset."""
    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the largest ocean?", "Pacific"),
        mock_example("What is 2+2?", "4"),
    ]
    knn = KNN(k=0, trainset=trainset, vectorizer=dspy.Embedder(DummyVectorizer()))
    query = {"question": "What is 3+3?"}
    nearest_samples = knn(**query)
    assert nearest_samples == [], "Expected empty list when k=0"


def test_knn_query_negative_k():
    """Tests that KNN with negative k returns an empty list."""
    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the largest ocean?", "Pacific"),
        mock_example("What is 2+2?", "4"),
    ]
    knn = KNN(k=-1, trainset=trainset, vectorizer=dspy.Embedder(DummyVectorizer()))
    query = {"question": "What is 3+3?"}
    nearest_samples = knn(**query)
    assert nearest_samples == [], "Expected empty list when k < 0"


def test_knn_query_k_exceeding_trainset():
    """Tests that KNN with k greater than trainset size returns all trainset examples."""
    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is 2+2?", "4"),
    ]
    knn = KNN(k=10, trainset=trainset, vectorizer=dspy.Embedder(DummyVectorizer()))
    query = {"question": "What is 3+3?"}
    nearest_samples = knn(**query)
    assert len(nearest_samples) == 2, "Expected all available examples when k exceeds trainset size"
