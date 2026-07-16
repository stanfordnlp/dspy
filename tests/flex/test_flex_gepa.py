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
from dspy.teleprompt.gepa.gepa_flex_utils import (
    enumerate_flex_submodules,
    flex_internal_predictor_ids,
    make_code_key,
)
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
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


def test_selection_eval_passes_program_trace_to_declaring_metric() -> None:
    """The selection eval (capture_traces=False) must pass the execution trace to a metric that
    declares `program_trace`, so a trace-dependent score (e.g. an LLM-call penalty that rewards
    deterministic code) actually drives candidate selection."""
    seen: dict[str, object] = {}

    def trace_aware_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        seen["trace"] = trace
        seen["n_calls"] = len(program_trace) if program_trace is not None else None
        return 1.0

    student = dspy.Flex(Echo)  # a flex submodule is present
    adapter = DspyAdapter(student_module=student, metric_fn=trace_aware_metric, feedback_map={})
    candidate = {make_code_key("self"): SIMPLE_MODULE}  # one dspy.Predict -> one traced call
    ex = dspy.Example(q="hi", a="hi").with_inputs("q")

    dspy.configure(lm=DummyLM([{"a": "hi"}]))
    adapter.evaluate([ex], candidate, capture_traces=False)  # the selection path

    assert seen.get("n_calls") == 1  # the metric received the trace, with the one call
    assert seen.get("trace") is None  # eval-mode semantics of the `trace` argument are preserved


def test_selection_eval_keeps_vanilla_semantics_for_legacy_metric() -> None:
    seen: dict[str, object] = {}

    def legacy_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        seen["trace"] = trace
        if trace is not None:
            return getattr(pred, "a", None) == gold.a  # bootstrapping mode: strict bool
        return 0.75  # eval mode: continuous score

    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=legacy_metric, feedback_map={})
    candidate = {make_code_key("self"): SIMPLE_MODULE}
    ex = dspy.Example(q="hi", a="hi").with_inputs("q")

    dspy.configure(lm=DummyLM([{"a": "hi"}]))
    batch = adapter.evaluate([ex], candidate, capture_traces=False)

    assert seen.get("trace") is None  # never fed the trace through the bootstrapping channel
    assert batch.scores == [0.75]


def test_selection_eval_binds_metric_with_required_contract_params() -> None:
    """A metric written to the full GEPAFeedbackMetric signature with `trace`/`pred_name`/
    `pred_trace` REQUIRED (no defaults) must still bind at flex scoring time: flex passes those
    declared parameters as None rather than dropping them, so the metric doesn't raise TypeError."""
    seen: dict[str, object] = {}

    def strict_metric(gold, pred, trace, pred_name, pred_trace, program_trace=None):
        seen["trace"] = trace
        seen["pred_name"] = pred_name
        seen["pred_trace"] = pred_trace
        seen["n_calls"] = len(program_trace) if program_trace is not None else None
        return 1.0

    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=strict_metric, feedback_map={})
    candidate = {make_code_key("self"): SIMPLE_MODULE}
    ex = dspy.Example(q="hi", a="hi").with_inputs("q")

    dspy.configure(lm=DummyLM([{"a": "hi"}]))
    batch = adapter.evaluate([ex], candidate, capture_traces=False)

    assert batch.scores == [1.0]  # bound and scored instead of raising TypeError
    assert seen.get("trace") is None and seen.get("pred_name") is None and seen.get("pred_trace") is None
    assert seen.get("n_calls") == 1  # program_trace still delivered to a metric that declares it


# --- adapter: scores stay aligned to the batch when examples crash ------------

CRASHY_MODULE = textwrap.dedent(
    """
    class CrashyModule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **inputs):
            if inputs["q"] == "boom":
                raise RuntimeError("runtime crash on this input")
            return dspy.Prediction(a=inputs["q"])
    """
).strip()


def test_evaluate_scores_stay_aligned_when_an_example_crashes() -> None:
    """A code candidate that binds fine but raises at runtime on one input gets dropped from
    bootstrap_trace_data's results. Scores must still come back one-per-example, aligned by
    position — the gepa engine pairs scores with example ids positionally, so a short list would
    credit example N+1's score to example N (and crash its state bookkeeping)."""

    def exact_match(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if getattr(pred, "a", None) == gold.a else 0.0

    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=exact_match, feedback_map={})
    candidate = {make_code_key("self"): CRASHY_MODULE}
    batch = [
        dspy.Example(q="hi", a="hi").with_inputs("q"),
        dspy.Example(q="boom", a="boom").with_inputs("q"),  # raises inside forward
        dspy.Example(q="yo", a="yo").with_inputs("q"),
    ]

    result = adapter.evaluate(batch, candidate, capture_traces=False)

    assert result.scores == [1.0, 0.0, 1.0]  # crashed example scores as a failure, in place
    assert result.outputs[1] is None  # no prediction for the crashed example
    assert getattr(result.outputs[2], "a", None) == "yo"  # later examples keep their own results


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
