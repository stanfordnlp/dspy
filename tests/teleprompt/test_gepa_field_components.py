"""Network-free tests using real signatures, adapters, GEPA and persistence."""

import json
from copy import deepcopy

import pytest
from pydantic import BaseModel, ValidationError, field_validator

import dspy
from dspy.teleprompt.gepa.gepa import DspyGEPAResult
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.teleprompt.gepa.signature_components import field_components, field_values

MARKER = "USE_CORRECT_ANSWER"


class Task(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField(desc="A question")
    answer: str = dspy.OutputField(desc="An answer")


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Task)

    def forward(self, question):
        return self.predict(question=question)


class FieldAwareEngine:
    """Deterministic model: the actual formatted description controls the answer."""

    def complete(self, request):
        from dspy.clients.lm15_boundary import request_kwargs
        from dspy.lm15 import Message, Response, Usage

        text = repr(request_kwargs(request, "chat")["messages"])
        answer = "good" if MARKER in text else "bad"
        return Response(
            None,
            request.model,
            Message.assistant(f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"),
            "stop",
            Usage(),
        )


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    assert pred_name in (None, "predict")
    return dspy.Prediction(score=float(pred.answer == gold.answer), feedback="Use the correct answer.")


def adapter(program=None, enabled=True):
    return DspyAdapter(
        student_module=program or Program(),
        metric_fn=metric,
        feedback_map={"predict": lambda **kwargs: {"score": 0.0, "feedback": "Fix answer"}},
        optimize_field_descriptions=enabled,
    )


def candidate(a):
    return DspyGEPAResult._candidate_components(a.student, a.optimize_field_descriptions)


def test_field_keys_are_explicit_and_default_unchanged():
    a = adapter(enabled=False)
    assert candidate(a) == {"predict": Task.instructions}
    assert a.get_adapter_state().keys() == {"rng_state"}
    b = adapter()
    assert len(candidate(b)) == 3
    assert set(field_values(b.student, b._field_components)) == set(candidate(b)) - {"predict"}


def test_immutable_fields_preserve_defaults_constraints_and_validators():
    class Nested(BaseModel):
        text: str

    class Typed(dspy.Signature):
        """Typed instructions."""

        question: str = dspy.InputField(desc="original", min_length=2)
        values: list[int] = dspy.InputField(default_factory=list)
        nested: Nested = dspy.OutputField()

        @field_validator("question")
        @classmethod
        def validate_question(cls, value):
            if value == "forbidden":
                raise ValueError("forbidden")
            return value

    p = dspy.Module()
    p.first = dspy.Predict(Typed)
    p.second = dspy.Predict(Typed)
    a = adapter(p)
    c = candidate(a)
    k = next(k for k, v in a._field_components.items() if v.predictor == "first" and v.field == "question")
    c[k] = "changed"
    out = a.build_program(c)
    new = out.first.signature
    assert new.fields["question"].json_schema_extra["desc"] == "changed"
    assert out.second.signature.fields["question"].json_schema_extra["desc"] == "original"
    assert Typed.fields["question"].json_schema_extra["desc"] == "original"
    assert new.fields["nested"].annotation is Nested
    assert new(question="ok", nested=Nested(text="x")).values == []
    for bad in ("x", "forbidden"):
        with pytest.raises(ValidationError):
            new(question=bad, nested=Nested(text="x"))
    assert list(new.fields) == list(Typed.fields)


def test_key_collision_is_rejected():
    p = Program()
    key = next(iter(field_components(p)))
    with pytest.raises(ValueError, match="collides"):
        field_components(p, [key])


@pytest.mark.parametrize("mutation", ["missing", "unknown-key", "nonstring"])
def test_invalid_candidate_rejected(mutation):
    a = adapter()
    c = candidate(a)
    if mutation == "missing":
        c.pop(next(iter(a._field_components)))
    elif mutation == "unknown-key":
        c["unknown"] = "x"
    else:
        c[next(iter(a._field_components))] = None
    with pytest.raises((TypeError, ValueError)):
        a.build_program(c)


def test_reflection_routes_fields_to_real_owner_trace():
    dspy.configure(lm=dspy.LM("dummy", engine=FieldAwareEngine(), cache=False), adapter=dspy.ChatAdapter())
    a = adapter()
    c = candidate(a)
    examples = [dspy.Example(question="q", answer="good").with_inputs("question")]
    batch = a.evaluate(examples, c, capture_traces=True)
    keys = list(a._field_components)
    result = a.make_reflective_dataset(c, batch, keys)
    assert set(result) == set(keys)
    for key, records in result.items():
        assert records[0]["Generated Outputs"]["answer"] == "bad"
        assert a._field_components[key].field in records[0]["Feedback"]
        assert "Return a field description" in records[0]["Feedback"]


def test_state_restore_checks_config_and_preserves_rng():
    a = adapter()
    a.rng.random()
    state = deepcopy(a.get_adapter_state())
    b = adapter()
    b.set_adapter_state(state)
    assert a.rng.random() == b.rng.random()
    with pytest.raises(ValueError, match="Cannot resume"):
        adapter(enabled=False).set_adapter_state(state)
    with pytest.raises(ValueError, match="Cannot resume"):
        b.set_adapter_state(adapter(enabled=False).get_adapter_state())
    p = Program()
    p.predict.signature = Task.with_updated_fields("question", desc="new seed")
    with pytest.raises(ValueError, match="Cannot resume"):
        adapter(p).set_adapter_state(state)


def test_native_compile_selects_field_mutation_saves_and_resumes(tmp_path):
    dspy.configure(lm=dspy.LM("dummy", engine=FieldAwareEngine(), cache=False), adapter=dspy.ChatAdapter())
    calls = []

    def proposer(candidate, reflective_dataset, components_to_update):
        calls.append(deepcopy(reflective_dataset))
        return {k: MARKER if k.startswith("@field:") else candidate[k] for k in components_to_update}

    data = [dspy.Example(question="q", answer="good").with_inputs("question")]

    def optimizer(enabled=True, budget=6):
        return dspy.GEPA(
            metric=metric,
            max_metric_calls=budget,
            instruction_proposer=proposer,
            optimize_field_descriptions=enabled,
            component_selector="all",
            skip_perfect_score=False,
            reflection_minibatch_size=1,
            num_threads=1,
            use_merge=False,
            track_stats=True,
            log_dir=str(tmp_path / "run"),
        )

    student = Program()
    out = optimizer().compile(student, trainset=data, valset=data)
    assert calls
    assert out(question="q").answer == "good"
    assert student(question="q").answer == "bad"
    assert out.detailed_results.val_aggregate_scores == [0.0, 1.0]
    serialized = out.detailed_results.to_dict()
    assert all(len(c) == 3 for c in serialized["candidates"])
    assert MARKER in json.dumps(serialized)
    out.save(tmp_path / "state.json")
    restored = Program()
    restored.load(tmp_path / "state.json")
    assert restored(question="q").answer == "good"
    out.save(tmp_path / "full", save_program=True)
    restored_full = dspy.load(tmp_path / "full", allow_pickle=True)
    assert restored_full(question="q").answer == "good"
    assert restored_full.detailed_results.to_dict()["candidates"] == serialized["candidates"]
    calls_before = len(calls)
    resumed = optimizer().compile(Program(), trainset=data, valset=data)
    assert resumed(question="q").answer == "good"
    assert len(calls) == calls_before
    assert resumed.detailed_results.total_metric_calls == out.detailed_results.total_metric_calls
    extended = optimizer(budget=9).compile(Program(), trainset=data, valset=data)
    assert len(calls) > calls_before
    assert extended.detailed_results.total_metric_calls > out.detailed_results.total_metric_calls
    assert extended(question="q").answer == "good"
    continuous_optimizer = optimizer(budget=9)
    continuous_optimizer.log_dir = str(tmp_path / "continuous")
    continuous = continuous_optimizer.compile(Program(), trainset=data, valset=data)
    assert continuous.detailed_results.to_dict()["candidates"] == extended.detailed_results.to_dict()["candidates"]
    assert continuous.detailed_results.val_aggregate_scores == extended.detailed_results.val_aggregate_scores
    assert continuous.detailed_results.total_metric_calls == extended.detailed_results.total_metric_calls
    with pytest.raises(ValueError, match="Cannot resume"):
        optimizer(enabled=False).compile(Program(), trainset=data, valset=data)


def test_auto_budget_counts_field_components(monkeypatch):
    counts = []

    def auto_budget(**kwargs):
        counts.append(kwargs["num_preds"])
        return 1

    def proposer(**kwargs):
        return {}

    dspy.configure(lm=dspy.LM("dummy", engine=FieldAwareEngine(), cache=False), adapter=dspy.ChatAdapter())
    data = [dspy.Example(question="q", answer="good").with_inputs("question")]
    for enabled in (False, True):
        opt = dspy.GEPA(
            metric=metric,
            auto="light",
            instruction_proposer=proposer,
            optimize_field_descriptions=enabled,
            num_threads=1,
            use_merge=False,
        )
        monkeypatch.setattr(opt, "auto_budget", auto_budget)
        opt.compile(Program(), trainset=data, valset=data)
    assert counts == [1, 3]


def test_builtin_proposer_receives_description_context():
    from dspy.utils.dummies import DummyLM

    a = adapter()
    a.reflection_lm = DummyLM([{"new_instruction": "Better field description"}])
    c = candidate(a)
    key = next(iter(a._field_components))
    records = {
        key: [
            {
                "Inputs": {"question": "q"},
                "Generated Outputs": {"answer": "bad"},
                "Feedback": "Optimize only input field 'question'. Return a field description.",
            }
        ]
    }
    result = a.propose_new_texts(c, records, [key])
    assert result == {key: "Better field description"}
    history = repr(a.reflection_lm.history)
    assert "A question" in history
    assert "Return a field description" in history


def test_nested_predictor_paths_and_chain_of_thought():
    p = dspy.Module()
    p.inner = dspy.Module()
    p.inner.cot = dspy.ChainOfThought(Task)
    a = adapter(p)
    c = candidate(a)
    assert len(c) == 4  # instructions plus question, reasoning, answer
    for key in a._field_components:
        c[key] = MARKER
    result = a.build_program(c)
    assert all(f.json_schema_extra["desc"] == MARKER for f in result.inner.cot.predict.signature.fields.values())
    assert all(f.json_schema_extra["desc"] != MARKER for f in p.inner.cot.predict.signature.fields.values())


def test_react_tool_objects_remain_unchanged():
    def search(query: str) -> str:
        """Original tool description."""
        return query

    p = dspy.ReAct("question -> answer", tools=[search])
    original = deepcopy(p.tools["search"].args)
    a = adapter(p)
    c = candidate(a)
    for key in a._field_components:
        c[key] = MARKER
    out = a.build_program(c)
    assert p.tools["search"].desc == out.tools["search"].desc == "Original tool description."
    assert p.tools["search"].args == out.tools["search"].args == original


@pytest.mark.parametrize("shared_stage", [False, True])
def test_field_reflection_does_not_mix_identical_predictors(shared_stage):
    class Twins(dspy.Module):
        def __init__(self):
            super().__init__()
            self.left = dspy.Predict(Task)
            self.right = self.left.deepcopy() if shared_stage else dspy.Predict(Task)

        def forward(self, question):
            self.left(question="left input")
            return self.right(question="right input")

    dspy.configure(lm=dspy.LM("dummy", engine=FieldAwareEngine(), cache=False), adapter=dspy.ChatAdapter())
    a = adapter(Twins())
    a.feedback_map = {name: lambda **kw: {"score": 0.0, "feedback": "Fix"} for name in ("left", "right")}
    c = candidate(a)
    data = [dspy.Example(question="q", answer="good").with_inputs("question")]
    batch = a.evaluate(data, c, capture_traces=True)
    result = a.make_reflective_dataset(c, batch, list(a._field_components))
    for key, component in a._field_components.items():
        assert result[key][0]["Inputs"]["question"] == f"{component.predictor} input"


def test_mixed_flex_code_and_predictor_fields_remain_separate():
    p = Program()
    p.flex = dspy.Flex("question -> answer")
    a = adapter(p)
    c = candidate(a)
    assert len(c) == 4
    assert "flex" in c
    assert all(component.predictor == "predict" for component in a._field_components.values())
    for key in a._field_components:
        c[key] = MARKER
    out = a.build_program(c)
    assert out.flex.module_src == p.flex.module_src
    assert out.flex.signature.fields["answer"].json_schema_extra["desc"] != MARKER
    assert out.predict.signature.fields["answer"].json_schema_extra["desc"] == MARKER


@pytest.mark.parametrize("change", ["annotation", "role", "prefix", "order"])
def test_resume_rejects_changed_field_contract(change):
    p = Program()
    if change == "order":
        p.predict.signature = dspy.Signature("a, b -> answer")
    state = adapter(p).get_adapter_state()
    if change == "annotation":
        p.predict.signature = Task.with_updated_fields("question", type_=int)
    elif change == "role":
        p.predict.signature = Task.with_updated_fields("question", __dspy_field_type="output")
    elif change == "prefix":
        p.predict.signature = Task.with_updated_fields("question", prefix="Renamed:")
    else:
        p.predict.signature = dspy.Signature("b, a -> answer", p.predict.signature.instructions)
    with pytest.raises(ValueError, match="Cannot resume"):
        adapter(p).set_adapter_state(state)


def test_full_program_save_retains_signature_validator(tmp_path):
    class Checked(dspy.Signature):
        value: str = dspy.InputField(default="ok")
        answer: str = dspy.OutputField()

        @field_validator("value")
        @classmethod
        def check_value(cls, value):
            if value == "bad":
                raise ValueError("bad value")
            return value

    p = dspy.Module()
    p.predict = dspy.Predict(Checked)
    a = adapter(p)
    c = candidate(a)
    for key in a._field_components:
        c[key] = MARKER
    out = a.build_program(c)
    out.save(tmp_path / "full", save_program=True)
    restored = dspy.load(tmp_path / "full", allow_pickle=True)
    assert restored.predict.signature(answer="a").value == "ok"
    with pytest.raises(ValidationError, match="bad value"):
        restored.predict.signature(value="bad", answer="a")


def test_callable_metadata_contract_survives_program_reconstruction():
    from typing import Annotated

    from pydantic import BeforeValidator

    def make_program():
        class Checked(dspy.Signature):
            question: list[Annotated[str, BeforeValidator(lambda value: str(value))]] = dspy.InputField()
            answer: str = dspy.OutputField()

        p = Program()
        p.predict.signature = Checked
        return p

    original = adapter(make_program())
    reconstructed = adapter(make_program())
    reconstructed.set_adapter_state(original.get_adapter_state())
    assert reconstructed.get_adapter_state() == original.get_adapter_state()


def test_callable_metadata_contract_is_stable_across_processes():
    import subprocess
    import sys

    script = """
import json
from typing import Annotated
from pydantic import BeforeValidator
import dspy
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
class Checked(dspy.Signature):
    question: list[Annotated[str, BeforeValidator(lambda value: str(value))]] = dspy.InputField()
    answer: str = dspy.OutputField()
p = dspy.Module()
p.predict = dspy.Predict(Checked)
a = DspyAdapter(p, lambda *a: 0, {}, optimize_field_descriptions=True)
print(json.dumps(a.get_adapter_state()['field_components_v1'], sort_keys=True))
"""
    first = subprocess.check_output([sys.executable, "-c", script], text=True)
    second = subprocess.check_output([sys.executable, "-c", script], text=True)
    assert json.loads(first) == json.loads(second)


@pytest.mark.parametrize("adapter_type", ["chat", "json"])
@pytest.mark.parametrize("include_failures", [False, True])
def test_real_adapter_parse_failure_reaches_field_reflection(adapter_type, include_failures):
    from dspy.teleprompt.bootstrap_trace import FailedPrediction

    class MalformedEngine:
        def complete(self, request):
            from dspy.lm15 import Message, Response, Usage

            return Response(None, request.model, Message.assistant("MALFORMED_RAW_OUTPUT"), "stop", Usage())

    chat_adapter = dspy.ChatAdapter(use_json_adapter_fallback=False) if adapter_type == "chat" else dspy.JSONAdapter()
    dspy.configure(lm=dspy.LM("dummy", engine=MalformedEngine(), cache=False), adapter=chat_adapter)
    a = adapter()
    a.add_format_failure_as_feedback = include_failures
    c = candidate(a)
    data = [dspy.Example(question="q", answer="good").with_inputs("question")]

    # No exception/FailedPrediction is fabricated: the real adapter parses the
    # engine's malformed completion and bootstrap_trace_data records its error.
    batch = a.evaluate(data, c, capture_traces=True)
    assert len(batch.outputs) == len(batch.trajectories) == 1
    assert isinstance(batch.outputs[0], FailedPrediction)
    assert batch.outputs[0].completion_text == "MALFORMED_RAW_OUTPUT"
    assert batch.scores == [0.0]
    traced_predictor, _, failure = batch.trajectories[0]["trace"][-1]
    assert traced_predictor._gepa_field_component_owner == "predict"
    assert isinstance(failure, FailedPrediction)

    keys = list(a._field_components)
    if not include_failures:
        with pytest.raises(Exception, match="No valid predictions"):
            a.make_reflective_dataset(c, batch, keys)
        return

    records = a.make_reflective_dataset(c, batch, keys)
    assert set(records) == set(keys)
    for key in keys:
        record = records[key][0]
        assert "MALFORMED_RAW_OUTPUT" in record["Generated Outputs"]
        assert "Your output failed to parse" in record["Feedback"]
        assert "Return a field description" in record["Feedback"]
        assert a._field_components[key].field in record["Feedback"]
