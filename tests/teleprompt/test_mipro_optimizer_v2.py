import random

import dspy
from dspy import Example
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.utils.dummies import DummyLM

trainset = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding").with_inputs("input"),
]


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


class SimpleModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("input -> output")

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def _compile_and_capture_seed(monkeypatch, optimizer, **compile_kwargs):
    """Run compile() with the expensive stages stubbed out and report the seed they received."""
    seen = {}

    def fake_bootstrap(self, program, trainset, seed, *args, **kwargs):
        seen["bootstrap"] = seed
        return None

    def fake_propose(self, *args, **kwargs):
        return {}

    def fake_optimize(
        self,
        program,
        instruction_candidates,
        demo_candidates,
        evaluate,
        valset,
        num_trials,
        minibatch,
        minibatch_size,
        minibatch_full_eval_steps,
        seed,
    ):
        seen["optimize"] = seed
        return program

    monkeypatch.setattr(MIPROv2, "_bootstrap_fewshot_examples", fake_bootstrap)
    monkeypatch.setattr(MIPROv2, "_propose_instructions", fake_propose)
    monkeypatch.setattr(MIPROv2, "_optimize_prompt_parameters", fake_optimize)

    optimizer.compile(SimpleModule(), trainset=trainset, minibatch=False, **compile_kwargs)
    return seen


def test_compile_honors_an_explicit_seed_of_zero(monkeypatch):
    dspy.configure(lm=DummyLM([{"output": "blue"}]))
    optimizer = MIPROv2(metric=simple_metric, auto=None, num_candidates=2, seed=9)

    seen = _compile_and_capture_seed(monkeypatch, optimizer, num_trials=2, seed=0)

    assert seen["bootstrap"] == 0
    assert seen["optimize"] == 0
    assert optimizer.rng.random() == random.Random(0).random()


def test_compile_falls_back_to_constructor_seed_when_unset(monkeypatch):
    dspy.configure(lm=DummyLM([{"output": "blue"}]))
    optimizer = MIPROv2(metric=simple_metric, auto=None, num_candidates=2, seed=9)

    seen = _compile_and_capture_seed(monkeypatch, optimizer, num_trials=2)

    assert seen["bootstrap"] == 9
    assert seen["optimize"] == 9
    assert optimizer.rng.random() == random.Random(9).random()
