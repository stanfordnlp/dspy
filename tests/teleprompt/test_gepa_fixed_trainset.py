from unittest import mock

import pytest

import dspy
from dspy.teleprompt.gepa.gepa import GEPA


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = dspy.Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


class DummyLM(dspy.clients.lm.LM):
    def __init__(self, history):
        super().__init__("dummy")
        self.history = history
        self.provider = "dummy"

    def __call__(self, prompt=None, messages=None, **kwargs):
        return ["dummy output"]


def simple_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return True


def test_gepa_fixed_trainset_too_large():
    fixed = [dspy.Example(input="f1").with_inputs("input"), dspy.Example(input="f2").with_inputs("input")]

    optimizer = GEPA(
        metric=simple_metric,
        fixed_trainset=fixed,
        reflection_minibatch_size=1,
        max_metric_calls=10,
        reflection_lm=DummyLM([]),
    )
    student = SimpleModule("input -> output")
    trainset = [dspy.Example(input="t1").with_inputs("input")]

    with pytest.raises(ValueError, match="cannot exceed reflection_minibatch_size"):
        optimizer.compile(student, trainset=trainset)


@mock.patch("gepa.optimize")
def test_gepa_fixed_trainset_sampler(mock_optimize):
    mock_gepa_result = mock.MagicMock()
    mock_gepa_result.best_candidate = {}
    mock_optimize.return_value = mock_gepa_result

    fixed = [dspy.Example(input="f1").with_inputs("input"), dspy.Example(input="f2").with_inputs("input")]
    trainset = [dspy.Example(input=f"t{i}").with_inputs("input") for i in range(10)]

    optimizer = GEPA(
        metric=simple_metric,
        fixed_trainset=fixed,
        reflection_minibatch_size=5,
        max_metric_calls=10,
        reflection_lm=DummyLM([]),
    )
    student = SimpleModule("input -> output")

    dspy.settings.configure(lm=DummyLM([]))

    optimizer.compile(student, trainset=trainset)

    mock_optimize.assert_called_once()
    kwargs = mock_optimize.call_args[1]

    assert "batch_sampler" in kwargs
    sampler = kwargs["batch_sampler"]
    assert sampler.num_fixed == 2
    assert sampler.minibatch_size == 5
    assert sampler.total_len == 12

    state = mock.MagicMock()
    state.total_metric_calls = 0
    state.i = 0

    # next_minibatch_ids twice
    batch1 = sampler.next_minibatch_ids(None, state)
    assert len(batch1) == 5
    assert batch1[0] == 0
    assert batch1[1] == 1
    assert 0 not in batch1[2:]
    assert 1 not in batch1[2:]

    state.i = 1
    batch2 = sampler.next_minibatch_ids(None, state)
    assert len(batch2) == 5
    assert batch2[0] == 0
    assert batch2[1] == 1
