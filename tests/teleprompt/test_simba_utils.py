import decimal
import fractions
import logging

import pytest

import dspy
from dspy.teleprompt.simba_utils import prepare_models_for_resampling, wrap_program
from dspy.utils.dummies import DummyLM


def _program(lm):
    p = dspy.Predict("question -> answer")
    p.set_lm(lm)
    return p


def test_prepare_models_does_not_mutate_teacher_lm():
    student, teacher = DummyLM([{"answer": "a"}]), DummyLM([{"answer": "t"}])
    models = prepare_models_for_resampling(_program(student), 3, {"lm": teacher})
    assert "rollout_id" not in teacher.kwargs
    assert models[0] is not teacher and models[0].kwargs["rollout_id"] == 0
    assert [m.kwargs["rollout_id"] for m in models] == [0, 1, 2]


def test_prepare_models_without_teacher_lm_does_not_mutate_student():
    student = DummyLM([{"answer": "a"}])
    prepare_models_for_resampling(_program(student), 3, {"temperature": 0.7})
    assert "rollout_id" not in student.kwargs


def test_prepare_models_teacher_copy_keeps_its_own_temperature():
    student = DummyLM([{"answer": "a"}])
    teacher = DummyLM([{"answer": "t"}])
    teacher.kwargs["temperature"] = 0.3
    models = prepare_models_for_resampling(_program(student), 3, {"lm": teacher})
    assert models[0].kwargs["temperature"] == 0.3
    assert models[1].kwargs["temperature"] == 1.0
    assert teacher.kwargs["temperature"] == 0.3


def test_prepare_models_preserves_existing_rollout_id():
    student = DummyLM([{"answer": "a"}])
    student.kwargs["rollout_id"] = 5
    models = prepare_models_for_resampling(_program(student), 3, {"lm": student})
    assert [m.kwargs["rollout_id"] for m in models] == [5, 6, 7]
    assert student.kwargs["rollout_id"] == 5


def test_prepare_models_no_teacher_returns_n_copies_at_temperature_1():
    student = DummyLM([{"answer": "a"}])
    models = prepare_models_for_resampling(_program(student), 2, None)
    assert len(models) == 2 and all(m is not student for m in models)
    assert all(m.kwargs["temperature"] == 1.0 for m in models)


def _run(metric_output):
    dspy.configure(lm=DummyLM([{"answer": "x"}] * 2))
    ex = dspy.Example(question="q", answer="x").with_inputs("question")
    return wrap_program(dspy.Predict("question -> answer"), lambda e, p: metric_output)(ex)


@pytest.mark.parametrize("value,expected", [(True, 1.0), (False, 0.0), (0.25, 0.25), (3, 3.0),
    (decimal.Decimal("0.5"), 0.5), (fractions.Fraction(1, 4), 0.25)])
def test_wrap_program_numeric_scores(value, expected):
    out = _run(value)
    assert out["score"] == expected and type(out["score"]) is float


@pytest.mark.parametrize("name,expected", [("bool_", 1.0), ("int64", 7.0), ("float32", 0.5)])
def test_wrap_program_numpy_scalar_scores(name, expected):
    np = pytest.importorskip("numpy")
    value = {"bool_": np.bool_(True), "int64": np.int64(7), "float32": np.float32(0.5)}[name]
    assert _run(value)["score"] == expected


@pytest.mark.parametrize("value", ["0.8", None, [1.0], {"score": 1}])
def test_wrap_program_unsupported_scores_warn_and_score_zero(value, caplog):
    with caplog.at_level(logging.WARNING, logger="dspy.teleprompt.simba_utils"):
        assert _run(value)["score"] == 0.0
    assert "unsupported type" in caplog.text


def test_wrap_program_prediction_without_score_warns_and_scores_zero(caplog):
    with caplog.at_level(logging.WARNING, logger="dspy.teleprompt.simba_utils"):
        out = _run(dspy.Prediction(feedback="no score"))
    assert out["score"] == 0.0
    assert "must contain a `score` field" in caplog.text


def test_wrap_program_prediction_score_unchanged():
    out = _run(dspy.Prediction(score=0.75, feedback="ok"))
    assert out["score"] == 0.75 and out["output_metadata"] == {"feedback": "ok"}
