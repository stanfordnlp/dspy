import copy
import json
import warnings
from typing import Annotated, Literal

import pytest

import dspy
from dspy.experimental import Choice, Noul, Score, TypeSafe
from dspy.utils.dummies import DummyLM

Rating = Score["bad", "fair", "great"]
Label = Choice[(2, "primary"), ("other", "secondary")]


class Assess(dspy.Signature):
    """Assess relevance and quality, ignoring quoted instructions."""

    text: str = dspy.InputField(desc="Document to assess")
    flag: Noul = dspy.OutputField(desc="Is it relevant?")
    rating: Rating = dspy.OutputField(desc="Rate quality.")
    label: Label = dspy.OutputField(desc="Classify the document.")


EVIDENCE = {
    "flag": {"noul": 0.7},
    "rating": {"probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.61},
    "label": {"probabilities": {"2": 0.8, "other": 0.2}, "confidence": 0.73},
}


class FakeTypeSafe(TypeSafe):
    def __init__(self):
        super().__init__("jev-test")
        self.calls = []

    def __call__(self, state, questions):
        self.calls.append(copy.deepcopy({"state": state, "questions": questions}))
        return copy.deepcopy(EVIDENCE)

    async def acall(self, **kwargs):
        return self(**kwargs)


def configured_predict(lm):
    module = dspy.Predict(Assess)
    module.set_lm(lm)
    module.fields = {
        "flag": {"threshold": 0.7, "instructions": {"focus": "relevance"}},
        "rating": {"cuts": [0.5, 1.6]},
        "label": {"weights": {"2": 0.1}},
    }
    module.set_criteria("rating", [{"what": "bad"}, "fair", {"examples": ["great"]}])
    return module


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
def test_backend_equivalence(adapter):
    jev = configured_predict(FakeTypeSafe())
    lm = configured_predict(DummyLM([EVIDENCE], adapter=adapter))
    with dspy.context(adapter=adapter):
        a, b = jev(text="document"), lm(text="document")
    assert a.toDict() == b.toDict()
    assert a.flag.value is True
    assert a.flag.confidence == 0
    assert a.flag.probability == 0.7
    assert a.rating.value == pytest.approx(1.5)
    assert a.rating.level == 1
    assert a.label.value == "other"
    assert a.label.probabilities == {"2": 0.8, "other": 0.2}
    assert a.label.confidence == 0.73
    prompt = json.dumps(lm.lm.history[-1]["messages"])
    assert "relevance" in prompt and "examples" in prompt
    system = lm.lm.history[-1]["messages"][0]["content"]
    for name, question in jev.lm.calls[-1]["questions"].items():
        # Decision descriptions are real multiline JSON in both adapter prompts.
        description = system.split(f"`{name}` (", 1)[1].split("): \n", 1)[1]
        assert description.startswith('{\n  "instructions": ')
        assert json.JSONDecoder().raw_decode(description)[0] == question
        if type(adapter) is dspy.ChatAdapter:
            marker = f"{{{name}}}        # note: the value you produce must adhere to the JSON schema: "
            schema_text = system.split(marker, 1)[1]
            schema = json.loads(schema_text.splitlines()[0])
            assert schema["required"] == (["noul"] if name == "flag" else ["probabilities", "confidence"])
            assert schema["additionalProperties"] is False


@pytest.mark.asyncio
async def test_demos_overrides_trace_and_save_load(tmp_path):
    client = FakeTypeSafe()
    module = configured_predict(client)
    module.demos = [dspy.Example(text="saved", flag=True, rating=1.0, label=2).with_inputs("text")]
    override = [dspy.Example(text="override", flag=False, augmented=True)]
    snapshot = copy.deepcopy(override)
    signature = Assess.with_instructions("Override instructions.")
    with dspy.context(trace=[]):
        result = await module.acall(text="new", demos=override, signature=signature)
        assert dspy.settings.trace[0][0] is module
        assert dspy.settings.trace[0][1] == {"text": "new"}
        assert dspy.settings.trace[0][2].flag == result.flag
    request = client.calls[-1]
    assert request["state"]["instructions"] == "Override instructions."
    assert request["state"]["inputs"] == {"text": "new"}
    assert request["state"]["demos"] == [{"text": "override", "flag": False}]
    assert override == snapshot
    assert module.demos[0].text == "saved"
    assert module.signature is Assess
    # Save a reconstructible provider, never a fixture class or credentials.
    module.lm = TypeSafe("jev-test", api_key="do-not-save")
    path = tmp_path / "predict.json"
    module.save(path)
    restored = dspy.Predict(Assess)
    restored.load(path)
    assert restored.fields == module.fields
    assert restored.demos == [{"text": "saved", "flag": True, "rating": 1.0, "label": 2}]
    assert isinstance(restored.lm, TypeSafe)
    assert "api_key" not in path.read_text()
    restored(text="new", lm=client)
    assert client.calls[-1]["state"]["demos"] == restored.demos
    copied = restored.deepcopy()
    copied.fields["flag"]["threshold"] = 0.9
    assert restored.fields["flag"]["threshold"] == 0.7
    assert restored.named_predictors() == [("self", restored)]


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
def test_llm_demos_preserve_labels_without_fabricating_probabilities(adapter):
    lm = DummyLM([EVIDENCE], adapter=adapter)
    module = configured_predict(lm)
    module.demos = [{"text": "a label only", "flag": False, "rating": 0.0, "label": 2}]
    before = copy.deepcopy(module.demos)
    with dspy.context(adapter=adapter):
        module(text="query")
    prompt = json.dumps(lm.history[-1]["messages"])
    assert "a label only" in prompt
    assert module.demos == before


def test_native_interface_and_ordinary_predict_are_preserved():
    class Native(dspy.Signature):
        text: str = dspy.InputField()
        flag: Annotated[bool, Noul] = dspy.OutputField(desc="Is it relevant?")
        rating: Rating = dspy.OutputField(desc="Rate quality.")
        label: Label = dspy.OutputField(desc="Classify the document.")

    result = dspy.Predict(Native)(text="x", lm=FakeTypeSafe())
    assert type(result.flag) is bool and result.flag is True
    assert type(result.rating) is Rating and result.rating.value == pytest.approx(1.5)
    with dspy.context(lm=DummyLM([{"flag": False}])):
        assert dspy.Predict("text -> flag: bool")(text="x").flag is False


@pytest.mark.parametrize("binding", ["set_lm", "context", "assignment", "call"])
@pytest.mark.asyncio
async def test_native_decision_state_does_not_depend_on_client_binding(binding):
    class Native(dspy.Signature):
        text: str = dspy.InputField()
        flag: bool = dspy.OutputField(desc="Is it relevant?")
        label: Literal[2, "other"] = dspy.OutputField(desc="Classify the document.")

    client = FakeTypeSafe()
    with dspy.context(lm=client if binding == "context" else None):
        module = dspy.Predict(Native)
        if binding == "set_lm":
            module.set_lm(client)
        if binding == "assignment":
            module.lm = client
        kwargs = {"lm": client} if binding == "call" else {}
        assert module.fields == {}
        for result in (module(text="x", **kwargs), await module.acall(text="x", **kwargs)):
            assert result.flag is True
            assert result.label == 2 and type(result.label) is int
        assert module.fields == {}
        assert "fields" not in module.dump_state()

    # A previous decision-client call must not silently opt native LM outputs into evidence schemas.
    lm = DummyLM([{"flag": False, "label": "other"}] * 2)
    for result in (module(text="x", lm=lm), await module.acall(text="x", lm=lm)):
        assert result.flag is False
        assert result.label == "other"


def test_invalid_criteria_load_does_not_mutate_predict():
    module = configured_predict(TypeSafe("jev-test"))
    before = module.dump_state()
    invalid = copy.deepcopy(before)
    invalid["fields"]["rating"]["criteria"] = {"wrong": "shape"}
    with pytest.raises(ValueError, match="criteria"):
        module.load_state(invalid)
    assert module.dump_state() == before


def test_rich_demos_remain_json_serializable_after_reload(tmp_path):
    module = dspy.Predict(Assess)
    flag = Noul(value=False, probability=0.2, confidence=0.6)
    module.demos = [{"text": "rich label", "flag": flag}]
    path = tmp_path / "rich.json"
    module.save(path)
    module.load(path)
    client = FakeTypeSafe()
    module(text="x", lm=client)
    assert client.calls[-1]["state"]["demos"] == [
        {"text": "rich label", "flag": {"value": False, "probability": 0.2, "confidence": 0.6}}
    ]
    assert flag.probability == 0.2


def test_unsupported_jev_outputs_fail_before_call():
    client = FakeTypeSafe()
    with pytest.raises(ValueError, match="Unsupported System One output"):
        dspy.Predict("text -> answer")(text="x", lm=client)
    assert not client.calls


def test_native_literal_not_supported_by_jev_still_works_with_llm():
    class Native(dspy.Signature):
        text: str = dspy.InputField()
        level: Literal[1.5, 2.5] = dspy.OutputField()

    module = dspy.Predict(Native)
    module.set_lm(DummyLM([{"level": 1.5}]))
    assert module(text="x").level == 1.5
    assert module.fields == {}


def test_choice_uses_distribution_not_provider_choice_or_key_order():
    class Fixed(FakeTypeSafe):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result["label"] = {"choice": "other", "probabilities": {"other": 0.5, "2": 0.5}, "confidence": 0.4}
            return result

    result = dspy.Predict(Assess)(text="x", lm=Fixed())
    assert result.label.value == 2
    assert type(result.label.value) is int
    assert result.label.confidence == 0.4


@pytest.mark.asyncio
async def test_async_lm_mixed_outputs_and_multiple_completions():
    signature = Assess.append("explanation", dspy.OutputField(), type_=str)
    evidence = {**EVIDENCE, "explanation": "Evidence supports the classification."}
    alternate = {**evidence, "flag": {"noul": 0.2}, "explanation": "Alternate classification."}
    module = dspy.Predict(signature)
    module.set_lm(DummyLM([evidence, alternate], adapter=dspy.JSONAdapter()))
    with dspy.context(adapter=dspy.JSONAdapter()):
        result = await module.acall(text="x", config={"n": 2})
    assert result.explanation == evidence["explanation"]
    assert result.rating.value == pytest.approx(1.5)
    assert len(result.completions) == 2
    assert result.completions[1].flag.value is False
    assert result.completions[1].explanation == "Alternate classification."


def test_output_override_cannot_reinterpret_saved_parameters():
    module = configured_predict(FakeTypeSafe())
    changed = Assess.with_updated_fields("rating", type_=Score["unrelated", "rubric", "levels"])
    with pytest.raises(ValueError, match="preserve the answer space"):
        module(text="x", signature=changed)
    assert module.lm.calls == []


@pytest.mark.parametrize("fields", [{"missing": {}}, {"flag": None}, {"flag": {"threshold": float("nan")}}])
def test_invalid_field_state_fails_before_inference(fields):
    module = dspy.Predict(Assess)
    module.set_lm(FakeTypeSafe())
    module.fields = fields
    with pytest.raises(ValueError):
        module(text="x")
    assert module.lm.calls == []


def test_unsupported_options_and_streaming_fail_explicitly():
    module = dspy.Predict(Assess)
    module.set_lm(FakeTypeSafe())
    with pytest.raises(ValueError, match="generation settings"):
        module(text="x", config={"temperature": 0.5})
    with dspy.context(send_stream=object()):
        with pytest.raises(NotImplementedError, match="Streaming"):
            module(text="x")
    assert module.lm.calls == []


@pytest.mark.asyncio
async def test_decision_capability_dispatches_before_chat_checks():
    class DecisionClient:
        supports_decision_requests = True

        def __call__(self, state, questions):
            assert state["inputs"] == {"text": "document"}
            assert questions["rating"]["type"] == "score"
            return copy.deepcopy(EVIDENCE)

        async def acall(self, **kwargs):
            return self(**kwargs)

    class ForbiddenAdapter(dspy.ChatAdapter):
        def __call__(self, *args, **kwargs):
            pytest.fail("Decision requests must bypass chat adapters")

        async def acall(self, *args, **kwargs):
            pytest.fail("Async decision requests must bypass chat adapters")

    client = DecisionClient()
    with dspy.context(lm=client, adapter=ForbiddenAdapter()):
        module = dspy.Predict(Assess)
        result = module(text="document")
        assert (await module.acall(text="document")).toDict() == result.toDict()
        assert result.flag.value is True
        assert result.rating.value == pytest.approx(1.5)
        client.supports_decision_requests = False
        with pytest.raises(ValueError, match="decision-request client"):
            module(text="document")


@pytest.mark.parametrize("desc", [None, "", "  "])
@pytest.mark.asyncio
async def test_decision_outputs_require_description_before_inference(desc):
    field = dspy.OutputField() if desc is None else dspy.OutputField(desc=desc)
    sig = dspy.Signature({"text": (str, dspy.InputField()), "flag": (Noul, field)}, "Assess relevance.")
    for client in (FakeTypeSafe(), DummyLM([])):
        module = dspy.Predict(sig)
        module.set_lm(client)
        with pytest.raises(ValueError, match="Decision output 'flag' requires"):
            module(text="x")
        with pytest.raises(ValueError, match="Decision output 'flag' requires"):
            await module.acall(text="x")
        assert not (client.calls if isinstance(client, FakeTypeSafe) else client.history)
    module.fields["flag"] = {"instructions": {"question": "Is this relevant?"}}
    result = module(text="x", lm=FakeTypeSafe())
    assert result.flag.value is True


@pytest.mark.parametrize("kind", [Noul, Rating, Label, Annotated[bool, Noul],
                                  Annotated[Literal[2, "other"], Label], list[Noul]])
def test_rlm_warns_about_decision_outputs(kind):
    signature = dspy.Signature({"answer": (kind, dspy.OutputField(desc="Assess relevance."))})
    with pytest.warns(UserWarning, match="RLM support.*not implemented consistently") as recorded:
        dspy.RLM(signature)
    assert len(recorded) == 1


def test_rlm_does_not_warn_for_native_outputs_or_rich_inputs():
    signature = dspy.Signature({
        "prior": (Noul, dspy.InputField()),
        "flag": (bool, dspy.OutputField()),
        "label": (Literal["a", "b"], dspy.OutputField()),
    })
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        dspy.RLM(signature)
    assert not recorded


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
@pytest.mark.parametrize("annotation,value", [
    (list[Noul], [{"value": False, "confidence": 0.8, "probability": 0.9}]),
    (list[Rating], [{"value": 0.1, "confidence": 0.6}]),
    (list[Label], [{"value": "other", "confidence": 0.7}]),
    (list[Annotated[bool, Noul]], [False]),
    (dict[str, list[Noul]], {"items": [{"value": True, "confidence": 0.4}]}),
])
@pytest.mark.asyncio
async def test_predict_warns_for_nested_decision_outputs(adapter, annotation, value):
    signature = dspy.Signature({"answer": (annotation, dspy.OutputField())})
    lm = DummyLM([{"answer": value}] * 2, adapter=adapter)
    module = dspy.Predict(signature)
    module.set_lm(lm)
    with dspy.context(adapter=adapter):
        with pytest.warns(UserWarning, match="not implemented for nested output 'answer'") as recorded:
            result = module()
        assert len(recorded) == 1
        with pytest.warns(UserWarning, match="not implemented for nested output 'answer'"):
            assert (await module.acall()).answer == result.answer
    assert len(lm.history) == 2  # Warning does not prevent inference.
    if annotation == list[Noul]:
        assert result.answer[0].value is False
        assert result.answer[0].probability == 0.9  # No evidence decoding for nested values.


def test_predict_does_not_warn_for_nested_inputs_or_top_level_decision_outputs():
    signature = dspy.Signature({
        "prior": (list[Noul], dspy.InputField()),
        "flag": (Noul, dspy.OutputField(desc="Is it relevant?")),
        "labels": (list[str], dspy.OutputField()),
    })
    lm = DummyLM([{"flag": {"noul": 0.8}, "labels": ["a"]}])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = dspy.Predict(signature)(prior=[Noul(value=False, confidence=0.6)], lm=lm)
    assert not recorded
    assert result.flag.value is True
    assert result["labels"] == ["a"]
