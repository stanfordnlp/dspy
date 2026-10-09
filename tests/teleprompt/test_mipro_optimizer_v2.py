import pytest

import dspy
from dspy import Example
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.utils.dummies import DummyLM

# A non-zero constructor seed, so falling back to it is distinguishable from
# honoring an explicit seed=0.
OPTIMIZER_SEED = 9


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


trainset = [
    Example(input="Question: What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="Question: What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs(
        "input"
    ),
]


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = dspy.Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


class _SeedCapturedError(Exception):
    """Raised by the patched `_set_random_seeds` to stop `compile` early."""


def _make_optimizer(**kwargs):
    lm = DummyLM([])
    return MIPROv2(metric=simple_metric, prompt_model=lm, task_model=lm, **kwargs)


def _resolve_seed(optimizer, **compile_kwargs):
    """Return the seed that `compile` resolves, without running the optimization.

    `_set_random_seeds` is the first thing `compile` does with the resolved seed,
    so intercepting it captures the value under test and lets us abort before
    bootstrapping, instruction proposal, or the optuna-backed parameter search.
    """
    captured = []

    def capture_seed(seed):
        captured.append(seed)
        raise _SeedCapturedError

    optimizer._set_random_seeds = capture_seed

    with pytest.raises(_SeedCapturedError):
        optimizer.compile(SimpleModule("input -> output"), trainset=trainset, **compile_kwargs)

    assert len(captured) == 1, "expected `_set_random_seeds` to be called exactly once"
    return captured[0]


def test_compile_honors_explicit_zero_seed():
    """seed=0 is a valid seed and must not be replaced by the optimizer default."""
    optimizer = _make_optimizer(seed=OPTIMIZER_SEED)
    assert _resolve_seed(optimizer, seed=0) == 0


def test_compile_honors_explicit_nonzero_seed():
    optimizer = _make_optimizer(seed=OPTIMIZER_SEED)
    assert _resolve_seed(optimizer, seed=42) == 42


def test_compile_falls_back_to_optimizer_seed_when_none():
    """None is the sentinel for "no override", so it falls back to the constructor seed."""
    optimizer = _make_optimizer(seed=OPTIMIZER_SEED)
    assert _resolve_seed(optimizer, seed=None) == OPTIMIZER_SEED


def test_compile_falls_back_to_optimizer_seed_when_omitted():
    optimizer = _make_optimizer(seed=OPTIMIZER_SEED)
    assert _resolve_seed(optimizer) == OPTIMIZER_SEED
