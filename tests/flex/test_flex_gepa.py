"""Tests for dspy.Flex code optimization inside dspy.GEPA.

Covers the behavior:
- a freshly constructed dspy.Flex binds a deterministic, LM-free baseline (a dspy.Module subclass;
  a single dspy.Predict when no tools are given),
- code-optimizable submodules are discovered by type (dspy.Flex), not by a duck-typed marker,
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
from dspy.teleprompt.bootstrap_trace import FailedPrediction
from dspy.teleprompt.gepa.gepa_flex_utils import enumerate_flex_submodules
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import LMRateLimitError

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


# --- baseline + discovery ----------------------------------------------------


def test_flex_baseline_is_predict_and_lm_free() -> None:
    # No LM configured: constructing a Flex must not make any LM call — with no tools it binds the
    # deterministic dspy.Predict baseline (a dspy.Module subclass).
    program = dspy.Flex(Echo)
    assert "dspy.Predict(" in program.module_src
    assert "q: str -> a: str" in program.module_src  # typed signature string
    assert "result.a" in program.module_src  # unwraps the declared output
    assert "class EchoModule(dspy.Module)" in program.module_src


def test_only_flex_instances_are_code_optimizable() -> None:
    """Discovery is by type. A module that merely looks flex-shaped is not rewritten: GEPA would
    hand it optimizer-authored source with nowhere safe to run it."""

    class LookAlike(dspy.Module):
        def _bind_code(self, src):
            raise AssertionError("must never be treated as a code component")

        def forward(self, **kwargs):
            return dspy.Prediction(a="x")

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.real = dspy.Flex(Echo)
            self.fake = LookAlike()

        def forward(self, **kwargs):
            return self.real(**kwargs)

    assert set(enumerate_flex_submodules(Prog())) == {"real"}


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
    assert set(flex_paths) == {"flex"}  # the regular Predict is not code-optimizable


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
    # Only the sibling Predict gets instruction optimization; the Flex's internal predictors are
    # owned by its code and never reported as tunable units — from the parent (Parameter leaf) or
    # from the Flex itself (named_predictors() is empty) — so no exclusion logic is needed.
    assert [n for n, _ in prog.named_predictors()] == ["sibling"]
    assert prog.flex.named_predictors() == []
    assert set(enumerate_flex_submodules(prog)) == {"flex"}


# --- adapter: build_program rebinds code, applies instructions ---------------


def test_build_program_rebinds_flex_code() -> None:
    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={})

    candidate = {"self": SIMPLE_MODULE}
    rebuilt = adapter.build_program(candidate)

    assert 'dspy.Predict("q -> a")' in rebuilt.module_src
    assert "self.p" in rebuilt.module_src
    assert "dspy.RLM(" not in rebuilt.module_src  # baseline replaced
    assert not hasattr(rebuilt, "p")  # predictors are per-forward artifacts, never set on the Flex


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
    candidate = {"self": SIMPLE_MODULE}  # one dspy.Predict -> one traced call
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
    candidate = {"self": SIMPLE_MODULE}
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
    candidate = {"self": SIMPLE_MODULE}
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
    """A code candidate that binds fine but raises at runtime on one input must still score
    one-per-example, aligned by position — the gepa engine pairs scores with example ids
    positionally, so a short list would credit example N+1's score to example N (and crash its
    state bookkeeping). The crashed slot carries a FailedPrediction with the error."""

    def exact_match(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if getattr(pred, "a", None) == gold.a else 0.0

    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=exact_match, feedback_map={})
    candidate = {"self": CRASHY_MODULE}
    batch = [
        dspy.Example(q="hi", a="hi").with_inputs("q"),
        dspy.Example(q="boom", a="boom").with_inputs("q"),  # raises inside forward
        dspy.Example(q="yo", a="yo").with_inputs("q"),
    ]

    result = adapter.evaluate(batch, candidate, capture_traces=False)

    assert result.scores == [1.0, 0.0, 1.0]  # crashed example scores as a failure, in place
    assert isinstance(result.outputs[1], FailedPrediction)  # the crash is captured, not dropped
    assert "runtime crash on this input" in result.outputs[1].completion_text
    assert getattr(result.outputs[2], "a", None) == "yo"  # later examples keep their own results


def test_crashed_examples_still_reach_code_reflective_records() -> None:
    """An example whose forward raised must reach the code proposer's reflective records with
    the failing input and the error — for a code candidate the crash IS the feedback. If it
    were dropped (bootstrap_trace_data's default), GEPA would score the slot as a failure yet
    reflection would never learn which input broke or how, and would keep proposing code with
    the same bug."""

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(Echo)  # a flex submodule routes evaluation through the trace path

        def forward(self, **kwargs):
            if kwargs["q"] == "boom":
                raise RuntimeError("runtime crash on this input")
            return dspy.Prediction(a=kwargs["q"])

    adapter = DspyAdapter(student_module=Prog(), metric_fn=_metric, feedback_map={})
    candidate = {"flex": SIMPLE_MODULE}
    batch = [
        dspy.Example(q="hi", a="hi").with_inputs("q"),
        dspy.Example(q="boom", a="boom").with_inputs("q"),  # raises inside forward
    ]

    eval_batch = adapter.evaluate(batch, candidate, capture_traces=True)
    assert eval_batch.scores == [1.0, 0.0]

    recs = adapter.make_reflective_dataset(candidate, eval_batch, ["flex"])["flex"]
    assert len(recs) == 2  # the crashed example is not silently dropped from reflection
    crashed = recs[1]
    assert crashed["Inputs"] == {"q": "boom"}  # the proposer sees which input broke
    assert "RuntimeError: runtime crash on this input" in str(crashed["Generated Outputs"])  # and how


# --- adapter: which build failures are absorbed ------------------------------


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("syntax error", "class Broken(dspy.Module)\n    def forward(self): pass"),
        ("no module class", "x = 1"),
        (
            "raises while constructing predictors",
            "class Broken(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.p = dspy.Nope('q -> a')\n"
            "    def forward(self, **inputs):\n"
            "        return dspy.Prediction(a='x')",
        ),
    ],
)
def test_a_candidate_that_cannot_bind_scores_as_a_failure(label, source) -> None:
    # Every way a candidate's own source can break binding — parsing it, running it, or the host
    # rejecting what it asks for — is the search's problem to score, not to crash on.
    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={}, failure_score=-1.0)
    ex = dspy.Example(q="hi", a="hi").with_inputs("q")

    result = adapter.evaluate([ex, ex], {"self": source}, capture_traces=False)

    assert result.scores == [-1.0, -1.0], label
    # No usable prediction either way: a bind failure yields None (the batch never ran); a
    # source that binds but breaks at forward yields a FailedPrediction carrying the error.
    assert all(o is None or isinstance(o, FailedPrediction) for o in result.outputs), label


def test_an_infrastructure_error_during_build_is_not_scored_as_a_failure(monkeypatch) -> None:
    """Only source-level failures are absorbed. An LM provider or rate-limit error is not the
    candidate's fault — swallowing it would silently score a whole run at the failure score and
    hand back an 'optimized' program chosen from garbage."""

    class RateLimitError(Exception):
        pass

    def boom(program, candidate):
        raise RateLimitError("429 from the provider")

    monkeypatch.setattr("dspy.teleprompt.gepa.gepa_utils.rebind_flex_code", boom)

    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={})
    ex = dspy.Example(q="hi", a="hi").with_inputs("q")

    with pytest.raises(RateLimitError):
        adapter.evaluate([ex], {"self": SIMPLE_MODULE}, capture_traces=False)


# --- adapter: propose routes code keys to the code proposer ------------------


def test_propose_new_texts_uses_code_proposer_for_code_keys() -> None:
    reflection = DummyLM([{"revised_source": SIMPLE_MODULE}])
    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={}, reflection_lm=reflection)
    ckey = "self"
    candidate = {ckey: student.module_src}
    reflective = {ckey: [{"Inputs": {"q": "x"}, "Generated Outputs": "wrong", "Feedback": "bad"}]}

    out = adapter.propose_new_texts(candidate, reflective, [ckey])
    assert 'dspy.Predict("q -> a")' in out[ckey]


def test_code_proposer_infra_error_propagates() -> None:
    """An LM infrastructure failure during code proposal is not the candidate's fault. Keeping
    the original source would make every iteration 'succeed' with an unchanged candidate and
    silently burn the optimization budget; propagating matches the instruction-proposer paths
    (and the build-time policy in DspyAdapter.evaluate)."""

    class DownLM(DummyLM):
        def forward(self, *args, **kwargs):
            raise LMRateLimitError("429 from the provider", model="dummy")

    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={}, reflection_lm=DownLM([]))
    candidate = {"self": student.module_src}
    reflective = {"self": [{"Inputs": {"q": "x"}, "Generated Outputs": "wrong", "Feedback": "bad"}]}

    with pytest.raises(LMRateLimitError):
        adapter.propose_new_texts(candidate, reflective, ["self"])


def test_code_proposer_bad_proposal_keeps_original() -> None:
    """A proposal-level failure that is not LM infrastructure (e.g. reflection output the
    proposal machinery can't handle) keeps that component's original source instead of
    crashing the run."""

    class GarbledLM(DummyLM):
        def forward(self, *args, **kwargs):
            raise RuntimeError("reflection output could not be handled")

    student = dspy.Flex(Echo)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={}, reflection_lm=GarbledLM([]))
    original = student.module_src
    candidate = {"self": original}
    reflective = {"self": [{"Inputs": {"q": "x"}, "Generated Outputs": "wrong", "Feedback": "bad"}]}

    out = adapter.propose_new_texts(candidate, reflective, ["self"])
    assert out == {"self": original}


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
    code_key = "flex"
    assert code_key in seed  # one code component for the whole Flex
    assert "class " in seed[code_key]  # the value is the full module class source
    assert "sibling" in seed  # the non-flex predictor's instruction
    # The Flex's internal predictors are NOT separate instruction components.
    assert not any(k.startswith("flex.") for k in seed)


# --- DspyGEPAResult.to_dict: candidates keep their code components -----------


def test_result_to_dict_keeps_flex_code_components() -> None:
    """to_dict() must serialize the same components GEPA optimized: the full module_src per Flex
    under its parameter path, and no entries for the Flex's internal predictors (whose
    instructions are written by that code, not optimized on their own)."""
    from dspy.teleprompt.gepa.gepa import DspyGEPAResult

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(Echo)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    prog = Prog()
    prog.flex._bind_code(SIMPLE_MODULE)

    result = DspyGEPAResult(
        candidates=[prog],
        parents=[[None]],
        val_aggregate_scores=[1.0],
        val_subscores=[{0: 1.0}],
        per_val_instance_best_candidates={0: {0}},
        discovery_eval_counts=[0],
    )
    (candidate,) = result.to_dict()["candidates"]

    code_key = "flex"
    assert candidate[code_key] == SIMPLE_MODULE  # the optimized code survives serialization
    assert candidate["sibling"] == prog.sibling.signature.instructions
    # The Flex's internal predictors are not components; reporting them would misstate what was
    # optimized and silently stand in for the dropped code.
    assert not any(k.startswith("flex.") for k in candidate)


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
