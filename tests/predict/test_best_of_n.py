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


def test_best_of_n_reusable_across_calls():
    """Test that BestOfN can be called multiple times with consistent fail_count behavior.

    Previously, self.fail_count was decremented during forward(), corrupting instance state
    and causing subsequent calls to tolerate fewer failures.
    """
    lm = DummyLM([
        {"answer": "a"}, {"answer": "b"}, {"answer": "c"},
        {"answer": "d"}, {"answer": "e"}, {"answer": "f"},
    ])
    dspy.configure(lm=lm)
    call_count = [0]

    def sometimes_fail(self, **kwargs):
        call_count[0] += 1
        # Fail on odd calls
        if call_count[0] % 2 == 1:
            raise ValueError("Odd call failure")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", sometimes_fail)
    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=2)

    # First call should succeed (1st attempt fails, 2nd succeeds)
    result1 = best_of_n(question="Call 1")
    assert result1 is not None, "First call should produce a result"

    # Reset call count
    call_count[0] = 0

    # Second call should behave identically — fail_count must NOT have been mutated
    result2 = best_of_n(question="Call 2")
    assert result2 is not None, "Second call should also produce a result (fail_count not corrupted)"


def test_best_of_n_all_failures_raises_error():
    """Test that BestOfN raises an error when all N attempts fail.

    Previously, forward() silently returned None when every attempt threw an exception.
    """
    lm = DummyLM([{"answer": "a"}] * 5)
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Always fails")

    predict = DummyModule("question -> answer", always_raise)
    # fail_count=N=5, so no early raise from the error threshold — we rely on the post-loop check
    best_of_n = BestOfN(module=predict, N=5, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=5)

    with pytest.raises(ValueError, match="Always fails"):
        best_of_n(question="Will all fail")

def test_best_of_n_zero_n():
    """Test that N=0 raises a ValueError in __init__."""
    predict = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    with pytest.raises(ValueError, match="N must be greater than 0"):
        BestOfN(module=predict, N=0, reward_fn=lambda _, __: 1.0, threshold=1.0)


def test_best_of_n_interleaved_success_and_failures():
    """Test that fail_count correctly tracks only failures, not all attempts.
    
    If N=5 and fail_count=3, and attempts 0, 2, and 3 fail while attempt 1 succeeds
    (but doesn't meet the threshold), the loop should continue to attempt 4
    because error_count=3 is not > fail_count=3.
    """
    lm = DummyLM([{"answer": "a"}] * 5)
    dspy.configure(lm=lm)
    call_count = [0]

    def interleaved_fail(self, **kwargs):
        current_call = call_count[0]
        call_count[0] += 1
        if current_call in [0, 2, 3]:
            raise ValueError(f"Intentional failure on call {current_call}")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", interleaved_fail)
    

    best_of_n = BestOfN(module=predict, N=5, reward_fn=lambda _, __: 0.0, threshold=1.0, fail_count=3)

    result = best_of_n(question="Interleaved")
    
    assert result is not None, "Should complete successfully and return the best prediction"
    assert call_count[0] == 5, "Should have made exactly 5 attempts"