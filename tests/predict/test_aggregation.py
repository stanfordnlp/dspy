import pytest

from dspy.evaluate import normalize_text
from dspy.predict.aggregation import majority
from dspy.primitives.prediction import Completions, Prediction


def test_majority_with_prediction():
    prediction = Prediction.from_completions([{"answer": "2"}, {"answer": "2"}, {"answer": "3"}])
    result = majority(prediction)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_completions():
    completions = Completions([{"answer": "2"}, {"answer": "2"}, {"answer": "3"}])
    result = majority(completions)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_list():
    completions = [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
    result = majority(completions)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_normalize():
    completions = [{"answer": "2"}, {"answer": " 2"}, {"answer": "3"}]
    result = majority(completions, normalize=normalize_text)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_field():
    completions = [
        {"answer": "2", "other": "1"},
        {"answer": "2", "other": "1"},
        {"answer": "3", "other": "2"},
    ]
    result = majority(completions, field="other")
    assert result.completions[0]["other"] == "1"


def test_majority_with_no_majority():
    completions = [{"answer": "2"}, {"answer": "3"}, {"answer": "4"}]
    result = majority(completions)
    assert result.completions[0]["answer"] == "2"  # The first completion is returned in case of a tie


def test_majority_with_prediction_without_completions():
    result = majority(Prediction(answer="Paris"))
    assert result.answer == "Paris"


def test_majority_empty_completions_raises_value_error():
    with pytest.raises(ValueError, match="at least one completion"):
        majority(Completions({}))
    with pytest.raises(ValueError, match="at least one completion"):
        majority(Prediction.from_completions([]))
    with pytest.raises(ValueError, match="at least one completion"):
        majority([])
