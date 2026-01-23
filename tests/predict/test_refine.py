import asyncio
import pytest

import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)

    async def aforward(self, **kwargs) -> Prediction:
        # Async version - just calls sync for DummyModule
        return self.forward_fn(self, **kwargs)


# ============== Existing sync tests ==============

def test_refine_forward_success_first_attempt():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    reward_call_count = [0]

    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)
    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = refine(question="What is the capital of Belgium?")

    assert result.answer == "Brussels"
    assert reward_call_count[0] > 0
    assert module_call_count[0] == 3


def test_refine_module_default_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)
    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)

    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")


def test_refine_module_custom_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def raise_on_second_call(self, **kwargs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", raise_on_second_call)
    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)

    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2


# ============== New async tests ==============

@pytest.mark.asyncio
async def test_refine_aforward_success_first_attempt():
    """Test async forward with sync reward function."""
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    reward_call_count = [0]

    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)
    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)

    with dspy.context(lm=lm):
        result = await refine.aforward(question="What is the capital of Belgium?")

    assert result.answer == "Brussels"
    assert reward_call_count[0] > 0
    assert module_call_count[0] == 3


@pytest.mark.asyncio
async def test_refine_aforward_with_async_reward():
    """Test async forward with async reward function."""
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    reward_call_count = [0]

    async def async_reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        await asyncio.sleep(0)  # Simulate async work
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)
    refine = Refine(module=predict, N=3, reward_fn=async_reward_fn, threshold=1.0)

    with dspy.context(lm=lm):
        result = await refine.aforward(question="What is the capital of Belgium?")

    assert result.answer == "Brussels"
    assert reward_call_count[0] > 0


@pytest.mark.asyncio
async def test_refine_aforward_default_fail_count():
    """Test async forward respects default fail count."""
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)
    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)

    with pytest.raises(ValueError):
        with dspy.context(lm=lm):
            await refine.aforward(question="What is the capital of Belgium?")


@pytest.mark.asyncio
async def test_refine_aforward_custom_fail_count():
    """Test async forward respects custom fail count."""
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    module_call_count = [0]

    def raise_on_second_call(self, **kwargs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", raise_on_second_call)
    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)

    with pytest.raises(ValueError):
        with dspy.context(lm=lm):
            await refine.aforward(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2


@pytest.mark.asyncio
async def test_refine_aforward_reaches_threshold():
    """Test async forward stops when threshold is reached."""
    # DummyLM needs enough responses for multiple rollout attempts
    lm = DummyLM([
        {"answer": "bad"}, {"answer": "bad"}, {"answer": "bad"},  # rollout 0
        {"answer": "good"}, {"answer": "good"}, {"answer": "good"},  # rollout 1
        {"answer": "also good"}, {"answer": "also good"}, {"answer": "also good"},  # rollout 2
    ])
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    def reward_fn(kwargs, pred: Prediction) -> float:
        return 1.0 if pred.answer == "good" else 0.0

    predict = DummyModule("question -> answer", count_calls)
    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)

    with dspy.context(lm=lm):
        result = await refine.aforward(question="Test?")

    assert result.answer == "good"