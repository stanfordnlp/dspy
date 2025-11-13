import pytest

import dspy
from dspy.teleprompt.knn_fewshot import KNNFewShot
from dspy.utils.dummies import DummyLM, DummyVectorizer


def mock_example(question: str, answer: str) -> dspy.Example:
    """Creates a mock DSP example with specified question and answer."""
    return dspy.Example(question=question, answer=answer).with_inputs("question")


@pytest.fixture
def setup_knn_few_shot() -> KNNFewShot:
    """Sets up a KNNFewShot instance for testing."""
    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the largest ocean?", "Pacific"),
        mock_example("What is 2+2?", "4"),
    ]
    return KNNFewShot(k=2, trainset=trainset, vectorizer=dspy.Embedder(DummyVectorizer()))


def test_knn_few_shot_initialization(setup_knn_few_shot):
    """Tests the KNNFewShot initialization."""
    knn_few_shot = setup_knn_few_shot
    assert knn_few_shot.KNN.k == 2, "Incorrect k value for KNN"
    assert len(knn_few_shot.KNN.trainset) == 3, "Incorrect trainset size for KNN"


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = dspy.Predict(signature)

    def forward(self, *args, **kwargs):
        return self.predictor(**kwargs)

    def reset_copy(self):
        # Creates a new instance of SimpleModule with the same predictor
        return SimpleModule(self.predictor.signature)


# TODO: Test not working yet
def _test_knn_few_shot_compile(setup_knn_few_shot):
    """Tests the compile method of KNNFewShot with SimpleModule as student."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")  # Assuming teacher uses the same module type

    # Setup DummyLM with a response for a query similar to one of the training examples
    lm = DummyLM(["Madrid", "10"])
    dspy.configure(lm=lm)  # Responses for the capital of Spain and the result of 5+5)

    knn_few_shot = setup_knn_few_shot
    trainset = knn_few_shot.KNN.trainset
    compiled_student = knn_few_shot.compile(student, teacher=teacher, trainset=trainset, valset=None)

    assert len(compiled_student.predictor.demos) == 1
    assert compiled_student.predictor.demos[0].input == trainset[0].input
    assert compiled_student.predictor.demos[0].output == trainset[0].output
    # Simulate a query that is similar to one of the training examples
    output = compiled_student.forward(input="What is the capital of Spain?").output

    # Validate that the output corresponds to one of the expected DummyLM responses
    # This assumes the compiled_student's forward method will execute the predictor with the given query
    assert output in ["Madrid", "10"], "The compiled student did not return the correct output based on the query"
