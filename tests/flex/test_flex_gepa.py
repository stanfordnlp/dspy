"""Tests for flex-marked (dspy.Flex) code optimization inside dspy.GEPA.

Covers the behavior:
- a freshly constructed dspy.Flex binds a deterministic, LM-free baseline (a dspy.Module subclass;
  a single dspy.Predict when no tools are given),
- the module is marked `_code_optimizable` and discoverable as a code-optimizable submodule,
- GEPA's seed candidate mixes per-submodule *code* components (each a full `module_src`) with
  *instruction* components for non-flex predictors, excluding predictors that live inside a Flex,
- the adapter rebinds Flex code from a candidate and routes code components through the
  code proposer,
- GEPA requires a non-empty trainset.
"""

from __future__ import annotations

import textwrap

import pytest

import dspy
from dspy.teleprompt.gepa.gepa_utils import (
    DspyAdapter,
    enumerate_flex_submodules,
    flex_internal_predictor_ids,
    make_code_key,
)
from dspy.utils.dummies import DummyLM

# A plain dspy.Predict module class that binds without an LM (no RLM interpreter needed).
SIMPLE_MODULE = textwrap.dedent(
    """
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict("q -> a")

        def forward(self, **inputs):
            r = self.p(q=inputs["q"])
            return dspy.Prediction(a=r.a)
    """
).strip()


class Echo(dspy.Signature):
    """Echo the question as the answer."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def _metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return 1.0 if getattr(pred, "a", None) == gold.a else 0.0


# --- baseline + marker -------------------------------------------------------


def test_flex_baseline_is_predict_and_lm_free() -> None:
    # No LM configured: constructing a Flex must not make any LM call — with no tools it binds the
    # deterministic dspy.Predict baseline (a dspy.Module subclass).
    program = dspy.Flex(Echo)
    assert "dspy.Predict(" in program.module_src
    assert "q: str -> a: str" in program.module_src  # typed signature string
    assert "result.a" in program.module_src  # unwraps the declared output
    assert "class EchoModule(dspy.Module)" in program.module_src


def test_flex_is_marked_code_optimizable() -> None:
    program = dspy.Flex(Echo)
    assert getattr(program, "_code_optimizable", False) is True


def test_enumerate_finds_top_level_and_nested_flex() -> None:
    top = dspy.Flex(Echo)
    assert set(enumerate_flex_submodules(top)) == {"self"}

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(Echo)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    prog = Prog()
    flex_paths = enumerate_flex_submodules(prog)
    assert set(flex_paths) == {"self.flex"}  # the regular Predict is not code-optimizable


# --- exclusion of Flex-internal predictors -----------------------------------


def test_flex_internal_predictors_excluded_from_instruction_components() -> None:
    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(Echo)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    prog = Prog()
    flex_subs = enumerate_flex_submodules(prog)
    internal_ids = flex_internal_predictor_ids(flex_subs)

    instruction_names = [n for n, p in prog.named_predictors() if id(p) not in internal_ids]
    # Only the sibling Predict gets instruction optimization; the Flex's internal
    # predictors are owned by its code and excluded.
    assert instruction_names == ["sibling"]
    all_names = [n for n, _ in prog.named_predictors()]
    assert any(n.startswith("flex.") for n in all_names)  # internals exist...
    assert not any(n.startswith("flex.") for n in instruction_names)  # ...but are excluded


# --- adapter: build_program rebinds code, applies instructions ---------------


def test_build_program_rebinds_flex_code() -> None:
    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={})

    candidate = {make_code_key("self"): SIMPLE_MODULE}
    rebuilt = adapter.build_program(candidate)

    assert 'dspy.Predict("q -> a")' in rebuilt.module_src
    assert "self.p" in rebuilt.module_src
    assert "dspy.RLM(" not in rebuilt.module_src  # baseline replaced
    assert hasattr(rebuilt, "p")  # the new predictor is attached flat on the module


# --- adapter: selection eval passes the trace to a flex metric ---------------


def test_selection_eval_passes_trace_to_flex_metric() -> None:
    """The selection eval (capture_traces=False) must pass the execution trace to the metric
    when a flex submodule is present, so a trace-dependent score (e.g. an LLM-call penalty that
    rewards deterministic code) actually drives candidate selection. GEPA's default scoring
    calls the metric in eval mode (trace=None), which would silently drop that signal."""
    seen: dict[str, object] = {}

    def trace_aware_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        seen["n_calls"] = len(trace) if trace is not None else None
        return 1.0

    student = dspy.Flex(Echo)  # a flex submodule is present
    adapter = DspyAdapter(student_module=student, metric_fn=trace_aware_metric, feedback_map={})
    candidate = {make_code_key("self"): SIMPLE_MODULE}  # one dspy.Predict -> one traced call
    ex = dspy.Example(q="hi", a="hi").with_inputs("q")

    dspy.configure(lm=DummyLM([{"a": "hi"}]))
    adapter.evaluate([ex], candidate, capture_traces=False)  # the selection path

    assert seen.get("n_calls") == 1  # the metric received the trace, with the one call


# --- adapter: propose routes code keys to the code proposer ------------------


def test_propose_new_texts_uses_code_proposer_for_code_keys() -> None:
    reflection = DummyLM([{"revised_source": SIMPLE_MODULE}])
    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={}, reflection_lm=reflection)
    ckey = make_code_key("self")
    candidate = {ckey: student.module_src}
    reflective = {ckey: [{"Inputs": {"q": "x"}, "Generated Outputs": "wrong", "Feedback": "bad"}]}

    out = adapter.propose_new_texts(candidate, reflective, [ckey])
    assert 'dspy.Predict("q -> a")' in out[ckey]


# --- GEPA.compile: seed mixes code + instruction components ------------------


class _FakeResult:
    def __init__(self, best_candidate):
        self.best_candidate = best_candidate
        self.val_aggregate_scores = []


def test_gepa_seed_mixes_code_and_instruction_components(monkeypatch) -> None:
    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(Echo)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    prog = Prog()
    captured = {}

    def fake_optimize(**kwargs):
        captured["seed"] = kwargs["seed_candidate"]
        return _FakeResult(best_candidate=kwargs["seed_candidate"])

    monkeypatch.setattr("gepa.optimize", fake_optimize)

    optimizer = dspy.GEPA(metric=_metric, reflection_lm=DummyLM([]), max_metric_calls=10)
    ex = dspy.Example(q="x", a="y", x="x", y="y").with_inputs("q", "x")
    optimizer.compile(prog, trainset=[ex], valset=[ex])

    seed = captured["seed"]
    code_key = make_code_key("self.flex")
    assert code_key in seed  # one code component for the whole Flex
    assert "class " in seed[code_key]  # the value is the full module class source
    assert "sibling" in seed  # the non-flex predictor's instruction
    # The Flex's internal predictors are NOT separate instruction components.
    assert not any(k.startswith("flex.") for k in seed)


# --- GEPA.compile: a non-empty trainset is required ------------------------------------


def test_gepa_flex_module_requires_trainset() -> None:
    # A Flex with no trainset must raise, like any other module — GEPA needs examples to
    # score candidates against.
    student = dspy.Flex(Echo)  # in-memory; baseline Predict
    optimizer = dspy.GEPA(metric=_metric, reflection_lm=DummyLM([]), max_metric_calls=10)
    with pytest.raises(AssertionError, match=r"[Tt]rainset"):
        optimizer.compile(student, trainset=[])


def test_gepa_plain_module_requires_trainset() -> None:
    class Plain(dspy.Module):
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict("q -> a")

        def forward(self, **kwargs):
            return self.p(**kwargs)

    optimizer = dspy.GEPA(metric=_metric, reflection_lm=DummyLM([]), max_metric_calls=10)
    with pytest.raises(AssertionError, match=r"[Tt]rainset"):
        optimizer.compile(Plain(), trainset=[])
