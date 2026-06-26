"""Tests for vibe-marked (dspy.Vibe) code optimization inside dspy.GEPA.

Covers the behavior:
- a freshly constructed dspy.Vibe binds a deterministic, LM-free dspy.RLM baseline (a
  dspy.Module subclass),
- the module is marked `_code_optimizable` and discoverable as a code-optimizable submodule,
- GEPA's seed candidate mixes per-submodule *code* components (each a full `module_src`) with
  *instruction* components for non-vibe predictors, excluding predictors that live inside a Vibe,
- the adapter rebinds Vibe code from a candidate and routes code components through the
  code proposer,
- GEPA with no trainset runs a bounded, unscored code-synthesis pass.

These deliberately avoid executing the RLM baseline (which needs a code interpreter):
optimized/synthesized candidates use plain dspy.Predict, which binds without an LM. Vibe
constructions pass check_intent=False so the (unrelated) intent check never makes an LM call.
"""

from __future__ import annotations

import textwrap

import pytest

import dspy
from dspy.teleprompt.gepa.gepa_utils import (
    DspyAdapter,
    enumerate_vibe_submodules,
    make_code_key,
    vibe_internal_predictor_ids,
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


def test_vibe_baseline_is_rlm_and_lm_free() -> None:
    # No LM configured: constructing a Vibe must not make any LM call — it binds the
    # deterministic dspy.RLM baseline (a dspy.Module subclass).
    program = dspy.Vibe(Echo, check_intent=False)
    assert "dspy.RLM(" in program.module_src
    assert "q: str -> a: str" in program.module_src  # typed signature string
    assert "result.a" in program.module_src  # unwraps the declared output
    assert "class EchoModule(dspy.Module)" in program.module_src


def test_vibe_is_marked_code_optimizable() -> None:
    program = dspy.Vibe(Echo, check_intent=False)
    assert getattr(program, "_code_optimizable", False) is True


def test_enumerate_finds_top_level_and_nested_vibe() -> None:
    top = dspy.Vibe(Echo, check_intent=False)
    assert set(enumerate_vibe_submodules(top)) == {"self"}

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.vibe = dspy.Vibe(Echo, check_intent=False)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.vibe(**kwargs)

    prog = Prog()
    vibe_paths = enumerate_vibe_submodules(prog)
    assert set(vibe_paths) == {"self.vibe"}  # the regular Predict is not code-optimizable


# --- exclusion of Vibe-internal predictors -----------------------------------


def test_vibe_internal_predictors_excluded_from_instruction_components() -> None:
    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.vibe = dspy.Vibe(Echo, check_intent=False)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.vibe(**kwargs)

    prog = Prog()
    vibe_subs = enumerate_vibe_submodules(prog)
    internal_ids = vibe_internal_predictor_ids(vibe_subs)

    instruction_names = [n for n, p in prog.named_predictors() if id(p) not in internal_ids]
    # Only the sibling Predict gets instruction optimization; the Vibe's internal
    # RLM predictors are owned by its code and excluded.
    assert instruction_names == ["sibling"]
    all_names = [n for n, _ in prog.named_predictors()]
    assert any(n.startswith("vibe.") for n in all_names)  # internals exist...
    assert not any(n.startswith("vibe.") for n in instruction_names)  # ...but are excluded


# --- adapter: build_program rebinds code, applies instructions ---------------


def test_build_program_rebinds_vibe_code() -> None:
    student = dspy.Vibe(Echo, check_intent=False)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={})

    candidate = {make_code_key("self"): SIMPLE_MODULE}
    rebuilt = adapter.build_program(candidate)

    assert 'dspy.Predict("q -> a")' in rebuilt.module_src
    assert "self.p" in rebuilt.module_src
    assert "dspy.RLM(" not in rebuilt.module_src  # baseline replaced
    assert hasattr(rebuilt, "p")  # the new predictor is attached flat on the module


def test_build_program_disables_vibe_auto_repair() -> None:
    student = dspy.Vibe(Echo, check_intent=False)
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={})
    candidate = {make_code_key("self"): SIMPLE_MODULE}
    rebuilt = adapter.build_program(candidate)
    # A broken proposed candidate must surface as an error during search, not silently
    # self-repair and pollute the persisted history.
    assert rebuilt._auto_repair is False


# --- adapter: propose routes code keys to the code proposer ------------------


def test_propose_new_texts_uses_code_proposer_for_code_keys() -> None:
    reflection = DummyLM([{"revised_source": SIMPLE_MODULE}])
    student = dspy.Vibe(Echo, check_intent=False)
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
            self.vibe = dspy.Vibe(Echo, check_intent=False)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.vibe(**kwargs)

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
    code_key = make_code_key("self.vibe")
    assert code_key in seed  # one code component for the whole Vibe
    assert "class " in seed[code_key]  # the value is the full module class source
    assert "sibling" in seed  # the non-vibe predictor's instruction
    # The Vibe's internal predictors are NOT separate instruction components.
    assert not any(k.startswith("vibe.") for k in seed)


# --- GEPA.compile: no-data code synthesis ------------------------------------


def test_gepa_no_data_runs_bounded_code_synthesis() -> None:
    student = dspy.Vibe(Echo, check_intent=False)  # in-memory; baseline RLM
    calls = {"metric": 0}

    def counting_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        calls["metric"] += 1
        return 1.0

    # Two synthesis rounds → two codegen responses (the CodegenSignature output is `module_src`).
    reflection = DummyLM([{"module_src": SIMPLE_MODULE}, {"module_src": SIMPLE_MODULE}])
    optimizer = dspy.GEPA(metric=counting_metric, reflection_lm=reflection, max_metric_calls=10)

    optimized = optimizer.compile(student, trainset=[])

    assert 'dspy.Predict("q -> a")' in optimized.module_src  # baseline was rewritten
    assert "dspy.RLM(" not in optimized.module_src
    assert calls["metric"] == 0  # no scoring without data


def test_gepa_no_data_without_vibe_module_requires_trainset() -> None:
    class Plain(dspy.Module):
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict("q -> a")

        def forward(self, **kwargs):
            return self.p(**kwargs)

    optimizer = dspy.GEPA(metric=_metric, reflection_lm=DummyLM([]), max_metric_calls=10)
    with pytest.raises(ValueError, match=r"[Tt]rainset"):
        optimizer.compile(Plain(), trainset=[])
