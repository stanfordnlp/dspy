"""Regression tests for https://github.com/stanfordnlp/dspy/issues/10321.

`MIPROv2.compile(seed=0)` must seed the run with 0; only `seed=None` falls
back to the constructor default. The heavy optimization steps are stubbed so
the tests run offline — they assert the resolved seed received downstream and
that identical seeds produce identical RNG states.
"""

import dspy


def _dummy_metric(example, pred, trace=None):
    return 1.0


class _TinySignature(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _make_optimizer(seed=9):
    lm = dspy.utils.DummyLM([{"answer": "42"}])
    return dspy.MIPROv2(metric=_dummy_metric, task_model=lm, prompt_model=lm, auto="light", seed=seed)


def _make_data():
    trainset = [dspy.Example(question=f"q{i}", answer="42").with_inputs("question") for i in range(4)]
    return trainset, list(trainset)


def _run_compile(seed, constructor_seed=9):
    optimizer = _make_optimizer(seed=constructor_seed)
    student = dspy.Predict(_TinySignature)
    trainset, valset = _make_data()
    captured = {}

    def fake_bootstrap(program, trainset, seed, *args, **kwargs):
        captured["bootstrap_seed"] = seed
        return []

    def fake_propose(*args, **kwargs):
        return []

    def fake_optimize(*args, **kwargs):
        captured["optimize_seed"] = kwargs["seed"] if "seed" in kwargs else args[-1]
        return student

    optimizer._bootstrap_fewshot_examples = fake_bootstrap
    optimizer._propose_instructions = fake_propose
    optimizer._optimize_prompt_parameters = fake_optimize
    optimizer.compile(student, trainset=trainset, valset=valset, minibatch=False, seed=seed)
    return captured, optimizer.rng.getstate()


def test_compile_seed_zero_is_honored():
    captured_a, state_a = _run_compile(seed=0)
    captured_b, state_b = _run_compile(seed=0)

    assert captured_a["bootstrap_seed"] == 0
    assert captured_a["optimize_seed"] == 0
    assert captured_b["bootstrap_seed"] == 0
    assert captured_b["optimize_seed"] == 0
    # Same seed reproduces the same RNG stream, distinct from the default.
    assert state_a == state_b
    _, state_default = _run_compile(seed=None)
    assert state_a != state_default


def test_compile_seed_none_falls_back_to_constructor_default():
    captured_a, state_a = _run_compile(seed=None)
    captured_b, state_b = _run_compile(seed=None)

    assert captured_a["bootstrap_seed"] == 9
    assert captured_a["optimize_seed"] == 9
    assert captured_b["bootstrap_seed"] == 9
    assert captured_b["optimize_seed"] == 9
    assert state_a == state_b


def test_compile_explicit_nonzero_seed_is_honored():
    captured, _ = _run_compile(seed=123)

    assert captured["bootstrap_seed"] == 123
    assert captured["optimize_seed"] == 123


def test_compile_seed_none_falls_back_to_custom_constructor_seed():
    captured, _ = _run_compile(seed=None, constructor_seed=42)

    assert captured["bootstrap_seed"] == 42
    assert captured["optimize_seed"] == 42
