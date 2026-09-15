"""Regression tests: BestOfN/Refine must not leak the failure budget across calls.

Before the fix, forward() decremented self.fail_count in place, so the
budget decayed monotonically over the life of the module: with
fail_count=1, call 1 tolerated one failure but call 3 raised on the very
first attempt. The budget must apply per call.

https://github.com/stanfordnlp/dspy/issues/10312
"""
import dspy
import pytest
from dspy.predict.best_of_n import BestOfN
from dspy.predict.refine import Refine
from dspy.predict.predict import Predict
from dspy.utils.dummies import DummyLM

INVOCATIONS = {"n": 0}


class CountingFailures(dspy.Module):
    """Every invocation raises; counts invocations so tests can measure
    how many attempts each forward() call consumed before raising."""

    def __init__(self):
        super().__init__()
        self.predictor = Predict("question -> answer")

    def forward(self, **kwargs):
        INVOCATIONS["n"] += 1
        raise ValueError("permanent failure")


@pytest.fixture(autouse=True)
def _reset_and_configure():
    INVOCATIONS["n"] = 0
    dspy.configure(lm=DummyLM([{"answer": "ok"}] * 40))


def _attempts_before_raise(make_module, forward_owner):
    """Call forward() once, count attempts consumed, expect the raise."""
    module = make_module()
    owner = forward_owner(module)
    before = INVOCATIONS["n"]
    with pytest.raises(ValueError, match="permanent failure"):
        owner.forward(question="x")
    return INVOCATIONS["n"] - before


def test_best_of_n_budget_identical_across_calls():
    def make():
        return CountingFailures()

    bon = BestOfN(module=make(), N=3, reward_fn=lambda _, __: 1.0,
                  threshold=0.0, fail_count=1)
    counts = [_attempts_before_raise(make, lambda m: bon) for _ in range(3)]
    # With fail_count=1 every call must tolerate one failure and raise on
    # the second attempt. The buggy version decayed the budget in place
    # (1 → 0 → -1), so call 3 raised on the FIRST attempt.
    assert counts == [2, 2, 2]


def test_refine_budget_identical_across_calls():
    def make():
        return CountingFailures()

    ref = Refine(module=make(), N=3, reward_fn=lambda _, __: 1.0,
                 threshold=0.0, fail_count=1)
    counts = [_attempts_before_raise(make, lambda m: ref) for _ in range(3)]
    assert counts == [2, 2, 2]
