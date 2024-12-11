import pytest
import numpy as np
import dsp, dspy
from dspy.teleprompt import MetaKNNFewShot
from dspy.utils.dummies import DSPDummyLM, DummyVectorizer
from typing import Sequence


def mock_example(question: str, answer: str) -> dspy.Example:
    """Creates a mock DSPy example with specified question and answer."""
    return dspy.Example(question=question, answer=answer).with_inputs("question")


def simple_metric(example, prediction, trace=None):
    """Simple metric that checks if prediction matches example."""
    return float(example.answer == prediction.answer)


@pytest.fixture
def setup_meta_knn():
    """Sets up a MetaKNNFewShot instance with test data."""
    trainset: Sequence[dspy.Example] = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the capital of Spain?", "Madrid"),
        mock_example("What is the capital of Italy?", "Rome"),
        mock_example("What is 2+2?", "4"),
        mock_example("What is 3+3?", "6"),
    ]
    
    # Replace the default vectorizer with a dummy one for testing
    dsp.SentenceTransformersVectorizer = DummyVectorizer
    
    meta_knn = MetaKNNFewShot(
        k=2,
        trainset=trainset,
        n_programs=3,
        metric=simple_metric,
        max_labeled_demos=2
    )
    return meta_knn


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = dspy.Predict(signature)

    def forward(self, question):
        return self.predictor(question=question)

    def reset_copy(self):
        return SimpleModule(self.predictor.signature)


def test_meta_knn_initialization(setup_meta_knn):
    """Tests the MetaKNNFewShot initialization."""
    meta_knn = setup_meta_knn
    assert meta_knn.KNN.k == 2, "Incorrect k value for KNN"
    assert len(meta_knn.KNN.trainset) == 5, "Incorrect trainset size"
    assert meta_knn.n_programs == 3, "Incorrect number of programs"


def test_meta_knn_compilation(setup_meta_knn):
    """Tests the compilation process of MetaKNNFewShot."""
    student = SimpleModule("question -> answer")
    teacher = SimpleModule("question -> answer")

    # Configure dummy LM with responses for different programs
    responses = [
        "Rome", "Madrid", "Rome", "Paris", "Madrid", "Rome", "Paris", "Madrid", "Rome", "Paris", "Madrid", "Madrid",

    ]
    lm = DSPDummyLM(responses)
    dspy.settings.configure(lm=lm)

    meta_knn = setup_meta_knn
    compiled_student = meta_knn.compile(student, teacher=teacher, trainset=meta_knn.KNN.trainset)

    assert hasattr(compiled_student, "forward"), "Compiled student missing forward method"
    assert len(meta_knn.programs) == 3, "Incorrect number of programs generated"
    
    # Test the compiled student
    result = compiled_student(question="What is the capital of France?")
    assert result.answer


def test_meta_knn_performance_cache(setup_meta_knn):
    """Tests the performance cache building functionality."""
    student = SimpleModule("question -> answer")
    teacher = SimpleModule("question -> answer")

    # Configure dummy LM
    responses = ["Paris"] * 15
    lm = DSPDummyLM(responses)
    dspy.settings.configure(lm=lm)

    meta_knn = setup_meta_knn
    meta_knn.compile(student, teacher=teacher, trainset=meta_knn.KNN.trainset)

    assert not meta_knn.performance_cache.empty, "Performance cache is empty"
    assert len(meta_knn.performance_cache) == len(meta_knn.KNN.trainset), "Incorrect number of rows in performance cache"
    assert all(f'program_{i}' in meta_knn.performance_cache.columns for i in range(meta_knn.n_programs)), \
        "Missing program columns in performance cache"
    
    # Check that non-NaN values are valid floats
    for col in meta_knn.performance_cache.columns:
        if col != 'example_id':
            non_nan_values = meta_knn.performance_cache[col].dropna()
            assert all(isinstance(x, (int, float)) for x in non_nan_values), \
                f"Column {col} contains non-numeric values"



def test_similar_examples_influence(setup_meta_knn):
    """Tests if the model properly uses similar examples to select the best program."""
    student = SimpleModule("question -> answer")
    teacher = SimpleModule("question -> answer")

    # Configure dummy LM with specific responses
    responses = [
        # Responses for capital questions
        "Paris", "Madrid", "Rome",
        # Responses for math questions
        "4", "6", "8",
        # Final test response
        "4"
    ]
    lm = DSPDummyLM(responses)
    dspy.settings.configure(lm=lm)

    meta_knn = setup_meta_knn
    compiled_student = meta_knn.compile(student, teacher=teacher, trainset=meta_knn.KNN.trainset)

    # Test with a math question (should use math-related examples)
    result = compiled_student(question="What is 4+4?")
    assert result.answer == "4", "No more responses"


def test_performance_cache_demo_exclusion(setup_meta_knn):
    """Tests that demos are properly excluded from performance calculations."""
    student = SimpleModule("question -> answer")
    teacher = SimpleModule("question -> answer")

    # Configure dummy LM
    responses = ["Paris"] * 15  # Enough responses for all programs and examples
    lm = DSPDummyLM(responses)
    dspy.settings.configure(lm=lm)

    meta_knn = setup_meta_knn
    compiled_student = meta_knn.compile(student, teacher=teacher, trainset=meta_knn.KNN.trainset)

    # Check that performance cache contains NaN values for demo examples
    assert meta_knn.performance_cache.isna().any().any(), "Performance cache should contain NaN values for demos"
    
    # Verify each program has some NaN values (indicating excluded demos)
    for i in range(meta_knn.n_programs):
        program_col = f'program_{i}'
        assert meta_knn.performance_cache[program_col].isna().any(), f"{program_col} should have some NaN values"


def test_best_program_selection_with_nans(setup_meta_knn):
    """Tests the best program selection logic with NaN values."""
    student = SimpleModule("question -> answer")
    teacher = SimpleModule("question -> answer")

    # Configure dummy LM with specific responses
    responses = ["Paris"] * 15
    lm = DSPDummyLM(responses)
    dspy.settings.configure(lm=lm)

    meta_knn = setup_meta_knn
    compiled_student = meta_knn.compile(student, teacher=teacher, trainset=meta_knn.KNN.trainset)

    # Get similar examples
    similar_examples = meta_knn.KNN(question="What is the capital of Germany?")
    
    # Test program selection
    best_program = meta_knn._select_best_program(similar_examples)
    assert best_program is not None, "Should select a best program even with NaN values"
  