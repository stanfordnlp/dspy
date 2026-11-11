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


def test_majority_with_non_string_values():
    completions = [{"answer": 4}, {"answer": 5}, {"answer": 4}]
    result = majority(completions)
    assert result.completions[0]["answer"] == 4

    completions = [{"answer": True}, {"answer": False}, {"answer": False}]
    result = majority(completions)
    assert result.completions[0]["answer"] is False


def test_majority_skips_none_values():
    completions = [{"answer": None}, {"answer": 7}, {"answer": None}]
    result = majority(completions)
    assert result.completions[0]["answer"] == 7


def test_majority_with_unhashable_values():
    completions_list = [{"answer": [1, 2]}, {"answer": [3]}, {"answer": [1, 2]}]
    result_list = majority(completions_list)
    assert result_list.completions[0]["answer"] == [1, 2]

    completions_dict = [{"answer": {"a": 1}}, {"answer": {"b": 2}}, {"answer": {"a": 1}}]
    result_dict = majority(completions_dict)
    assert result_dict.completions[0]["answer"] == {"a": 1}

    completions_tie = [{"answer": ["first"]}, {"answer": ["second"]}]
    result_tie = majority(completions_tie)
    assert result_tie.completions[0]["answer"] == ["first"]

    completions_mixed = [{"answer": 1}, {"answer": 1}, {"answer": [2]}, {"answer": [2]}]
    result_mixed = majority(completions_mixed, normalize=None)
    assert result_mixed.completions[0]["answer"] == 1


def test_majority_with_numpy_arrays():
    import numpy as np

    completions = [
        {"answer": np.array([1, 2])},
        {"answer": np.array([3, 4])},
        {"answer": np.array([1, 2])},
    ]

    result = majority(completions, normalize=None)

    np.testing.assert_array_equal(result.completions[0]["answer"], np.array([1, 2]))
