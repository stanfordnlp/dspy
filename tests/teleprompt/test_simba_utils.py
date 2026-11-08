import dspy
from dspy.teleprompt.simba_utils import prepare_models_for_resampling
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
