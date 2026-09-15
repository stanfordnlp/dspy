import dspy
from dspy.teleprompt.simba import SIMBA
from dspy.teleprompt.simba_utils import prepare_models_for_resampling
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


def test_prepare_models_for_resampling_default():
    lm = DummyLM([])
    dspy.configure(lm=lm)
    program = DummyModule()

    models = prepare_models_for_resampling(program, n=3)
    assert len(models) == 3
    assert [m.kwargs.get("temperature") for m in models] == [1.0, 1.0, 1.0]
    assert [m.kwargs.get("rollout_id") for m in models] == [0, 1, 2]


def test_prepare_models_for_resampling_single_temp():
    lm = DummyLM([])
    dspy.configure(lm=lm)
    program = DummyModule()

    models = prepare_models_for_resampling(program, n=3, temperatures=0.7)
    assert len(models) == 3
    assert [m.kwargs.get("temperature") for m in models] == [0.7, 0.7, 0.7]


def test_prepare_models_for_resampling_list_temp():
    lm = DummyLM([])
    dspy.configure(lm=lm)
    program = DummyModule()

    models = prepare_models_for_resampling(program, n=5, temperatures=[0.3, 0.7, 1.0])
    assert len(models) == 5
    assert [m.kwargs.get("temperature") for m in models] == [0.3, 0.7, 1.0, 0.3, 0.7]


def test_prepare_models_for_resampling_with_teacher():
    lm = DummyLM([])
    teacher_lm = DummyLM([])
    dspy.configure(lm=lm)
    program = DummyModule()

    models = prepare_models_for_resampling(
        program, n=3, teacher_settings={"lm": teacher_lm}, temperatures=[0.5, 0.8]
    )
    assert len(models) == 3
    assert models[0] == teacher_lm
    assert [m.kwargs.get("temperature") for m in models[1:]] == [0.5, 0.8]


def test_simba_init_candidate_temperatures():
    simba = SIMBA(metric=lambda x, y: 1.0, candidate_temperatures=[0.5, 1.0])
    assert simba.candidate_temperatures == [0.5, 1.0]
