import pytest

from dspy.primitives.prediction import Completions, Prediction


def test_completions_len():
    completions = Completions([{"answer": "2"}, {"answer": "3"}])
    assert len(completions) == 2


def test_empty_completions_len_is_zero():
    assert len(Completions([])) == 0
    assert len(Completions({})) == 0


def test_empty_completions_is_falsy_and_iterates_empty():
    completions = Completions([])
    assert not completions
    assert list(completions) == []
    with pytest.raises(IndexError):
        completions[0]


def test_prediction_repr_with_empty_completions():
    prediction = Prediction.from_completions([])
    assert repr(prediction) == "Prediction(\n    \n)"


def test_prediction_repr_with_multiple_completions():
    prediction = Prediction.from_completions([{"answer": "2"}, {"answer": "3"}])
    assert "1 completions omitted" in repr(prediction)
