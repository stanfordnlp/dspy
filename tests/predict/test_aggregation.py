from dspy.predict.aggregation import majority
from dspy.primitives.prediction import Prediction, Completions
from dsp.utils import normalize_text
from dspy.utils.dummies import DummySignature, make_dummy_completions


def test_majority_with_prediction():
    completions = make_dummy_completions(
        DummySignature, [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
    )
    prediction = Prediction.from_completions(completions)

    result = prediction.get_majority()
    assert result.completions[0]["answer"] == "2"


def test_majority_with_completions():
    prediction = Prediction.from_completions(
        make_dummy_completions(
            DummySignature, [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
        )
    )
    result = majority(prediction)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_list():
    prediction = Prediction.from_completions(
        make_dummy_completions(
            DummySignature, [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
        )
    )
    result = majority(prediction)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_normalize():
    prediction = Prediction.from_completions(
        make_dummy_completions(
            DummySignature, [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
        )
    )
    result = majority(prediction, normalize=normalize_text)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_field():
    prediction = Prediction.from_completions(
        make_dummy_completions(
            DummySignature,
            [
                {"answer": "2", "other": "1"},
                {"answer": "2", "other": "1"},
                {"answer": "3", "other": "2"},
            ],
        )
    )
    result = majority(prediction, field="other")
    assert result.completions[0]["other"] == "1"


def test_majority_with_no_majority():
    prediction = Prediction.from_completions(
        make_dummy_completions(
            DummySignature, [{"answer": "2"}, {"answer": "3"}, {"answer": "4"}]
        )
    )
    result = majority(prediction)
    assert (
        result.completions[0]["answer"] == "2"
    )  # The first completion is returned in case of a tie
