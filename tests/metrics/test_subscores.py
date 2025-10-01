import math

import pytest

import dspy
from dspy.metrics import subscore
from dspy.metrics._subscores import _begin_collect, _end_collect, finalize_scores


def test_subscore_expression_resolution():
    token = _begin_collect()
    try:
        acc = subscore("acc", 0.8, bounds=(0, 1))
        lat = subscore("latency", 0.2, maximize=False, units="s")
        result = acc - 0.5 * lat
    finally:
        collector = _end_collect(token)

    scores = finalize_scores(result, collector)
    assert math.isclose(scores.scalar, 0.8 - 0.1)
    assert scores.subscores == {"acc": 0.8, "latency": 0.2}
    assert "expr" in scores.info and "latency" in scores.info["meta"]


def test_subscore_float_cast_keeps_subscores():
    token = _begin_collect()
    try:
        acc = subscore("acc", 1.0)
        value = float(acc)
    finally:
        collector = _end_collect(token)

    scores = finalize_scores(value, collector)
    assert scores.subscores == {"acc": 1.0}
    assert scores.info["expr"] is None


def test_subscore_duplicate_name_raises():
    token = _begin_collect()
    try:
        subscore("acc", 1.0)
        with pytest.raises(ValueError):
            subscore("acc", 0.5)
    finally:
        _end_collect(token)


def test_prediction_callable_score_resolution():
    example = dspy.Example(question="What is 1+1?", answer="2").with_inputs("question")
    prediction = dspy.Prediction(answer="2")

    def score_fn(ex, pred, ctx=None):
        assert ex.answer == "2"
        return subscore("acc", 1.0)

    prediction.score = score_fn
    prediction.bind_example(example)

    with pytest.raises(TypeError):
        float(prediction)

    scores = prediction.resolve_score()
    assert isinstance(scores.subscores, dict) and scores.subscores["acc"] == 1.0
    assert math.isclose(prediction.score, 1.0)
    assert math.isclose(float(prediction), 1.0)
