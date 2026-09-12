import pytest

import dspy
from dspy.predict.best_of_n import BestOfN
from dspy.predict.predict import Predict
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


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
        # The answer should always be one word.
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)

    best_of_n = BestOfN(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = best_of_n(question="What is the capital of Belgium?")

    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] > 0, "Reward function should have been called"
    assert module_call_count[0] == 3, (
        "Module should have been called exactly 3 times, but was called %d times" % module_call_count[0]
    )


def test_refine_module_default_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)

    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")


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

    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )


def test_best_of_n_fail_count_resets_across_calls():
    """fail_count is a per-call retry budget; the same instance must tolerate
    the same number of failures on every call (regression for mutating
    self.fail_count, which eroded the budget across calls)."""
    lm = DummyLM([{"answer": "Brussels"}] * 10)
    dspy.configure(lm=lm)
    module_call_count = [0]

    def always_raise(self, **kwargs):
        module_call_count[0] += 1
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)
    best_of_n = BestOfN(module=predict, N=5, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=2)

    # fail_count=2: tolerate 2 failures, raise on the 3rd attempt.
    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 3, (
        "First call should make 3 attempts (2 tolerated + 1 raised), got %d" % module_call_count[0]
    )

    # The second call to the same instance must behave identically.
    module_call_count[0] = 0
    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 3, (
        "Second call should also make 3 attempts, got %d (budget leaked across calls)" % module_call_count[0]
    )


def test_best_of_n_fail_count_zero_raises_on_first_failure():
    """fail_count=0 means zero allowed failures: the first failed attempt must
    raise immediately (and 0 must not be swallowed by `fail_count or N`)."""
    lm = DummyLM([{"answer": "Brussels"}] * 10)
    dspy.configure(lm=lm)
    module_call_count = [0]

    def always_raise(self, **kwargs):
        module_call_count[0] += 1
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)
    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=0)

    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 1, (
        "fail_count=0 should raise on the first failure, got %d attempts" % module_call_count[0]
    )
