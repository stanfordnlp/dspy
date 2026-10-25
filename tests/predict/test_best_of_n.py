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
    module_call_count = [0]

    def always_raise(self, **kwargs):
        module_call_count[0] += 1
        raise ValueError(f"Failure {module_call_count[0]}")

    predict = DummyModule("question -> answer", always_raise)

    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError, match="Failure 3"):
        best_of_n(question="What is the capital of Belgium?")
    with pytest.raises(ValueError, match="Failure 6"):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 6
    assert best_of_n.fail_count == 3


def test_refine_module_zero_fail_count():
    lm = DummyLM([{"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def always_raise(self, **kwargs):
        module_call_count[0] += 1
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)
    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=0)

    with pytest.raises(ValueError, match="Deliberately failing"):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 1
    assert best_of_n.fail_count == 0


def test_refine_module_succeeds_after_allowed_failure():
    lm = DummyLM([{"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def raise_once(self, **kwargs):
        module_call_count[0] += 1
        if module_call_count[0] == 1:
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", raise_once)
    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)

    result = best_of_n(question="What is the capital of Belgium?")
    assert result.answer == "Brussels"
    assert module_call_count[0] == 2


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
