import json
from typing import Annotated, Literal

import pytest
from pydantic import Field, TypeAdapter, ValidationError

import dspy
from dspy.adapters.decision import evidence_type, resolve_adapter
from dspy.adapters.json_adapter import _get_structured_outputs_response_format
from dspy.experimental import Choice, Noul, Score
from dspy.utils.dummies import DummyLM
from tests.predict.test_decision_parameters import FakeClient

Severity = Score["Minor", "Disruptive", "Blocking"]
Category = Choice[("billing", "Payment issue"), ("technical", "Product malfunction")]


@pytest.mark.parametrize("left,right,left_labels,right_labels", [
    (Category, Choice[("a", "A"), ("b", "B")], ["billing", "technical"], ["a", "b"]),
    (Severity, Score["Low", "High"], ["0", "1", "2"], ["0", "1"]),
])
def test_short_evidence_names_keep_distinct_answer_spaces(left, right, left_labels, right_labels):
    assert evidence_type(left).__name__ == evidence_type(right).__name__
    sig = dspy.Signature({
        "first": (evidence_type(left), dspy.OutputField()),
        "second": (evidence_type(right), dspy.OutputField()),
    })
    model = _get_structured_outputs_response_format(sig)
    schema = model.model_json_schema()
    for field, labels in (("first", left_labels), ("second", right_labels)):
        evidence = schema["$defs"][schema["properties"][field]["$ref"].split("/")[-1]]
        probabilities = schema["$defs"][evidence["properties"]["probabilities"]["$ref"].split("/")[-1]]
        assert list(probabilities["properties"]) == labels
    first = {"probabilities": dict.fromkeys(left_labels, 1 / len(left_labels)), "confidence": 0.7}
    second = {"probabilities": dict.fromkeys(right_labels, 1 / len(right_labels)), "confidence": 0.6}
    model.model_validate({"first": first, "second": second})
    with pytest.raises(ValidationError):
        model.model_validate({"first": second, "second": first})


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
def test_ordinary_annotated_fields_keep_existing_parsing_and_schema(adapter):
    sig = dspy.Signature({
        "count": (Annotated[int, Field(gt=0)], dspy.OutputField()),
        "name": (Annotated[str, Field(max_length=3)], dspy.OutputField()),
    })
    values = {"count": -1, "name": "longer"}
    lm = DummyLM([values], adapter=adapter)
    with dspy.context(lm=lm, adapter=adapter):
        assert dspy.Predict(sig)().toDict() == values
    properties = _get_structured_outputs_response_format(sig).model_json_schema()["properties"]
    assert "exclusiveMinimum" not in properties["count"]
    assert "maxLength" not in properties["name"]


def test_decision_apis_are_experimental_only():
    from dspy.experimental import TypeSafe

    for api in (Choice, Noul, Score, TypeSafe):
        assert getattr(dspy.experimental, api.__name__) is api
        assert not hasattr(dspy, api.__name__)
        assert "Experimental:" in api.__doc__


def decision_fields(rich):
    return {
        "urgent": Noul if rich else bool,
        "severity": Severity,
        "category": Category if rich else Literal["billing", "technical"],
    }


def decision_signature(rich, direction):
    descriptions = {"urgent": "Is it urgent?", "severity": "Rate severity.", "category": "Classify the issue."}
    fields = {
        name: (kind, dspy.InputField() if direction == "input" else dspy.OutputField(desc=descriptions[name]))
        for name, kind in decision_fields(rich).items()
    }
    if direction == "input":
        fields["accept"] = (bool, dspy.OutputField(desc="Accept the assessment?"))
    else:
        fields = {"ticket": (str, dspy.InputField()), **fields}
    return dspy.Signature(fields, "Assess the ticket.")


def decision_values(rich, evidence=False):
    return {
        "urgent": Noul(value=True, confidence=0.6, **({"probability": 0.8} if evidence else {})) if rich else True,
        "severity": Severity(
            value=1.5, confidence=0.61, **({"probabilities": {0: 0.1, 1: 0.3, 2: 0.6}} if evidence else {})
        ),
        "category": Category(
            value="technical",
            confidence=0.73,
            **({"probabilities": {"billing": 0.2, "technical": 0.8}} if evidence else {}),
        ) if rich else "technical",
    }


def decision_evidence(rich):
    return {
        "urgent": {"noul": 0.8} if rich else True,
        "severity": {"probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.61},
        "category": (
            {"probabilities": {"billing": 0.2, "technical": 0.8}, "confidence": 0.73} if rich else "technical"
        ),
    }


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
@pytest.mark.parametrize("rich", [False, True])
def test_predict_outputs_and_generated_schema(adapter, rich):
    signature = decision_signature(rich, "output")
    values = decision_values(rich, evidence=True)
    with dspy.context(lm=DummyLM([decision_evidence(rich)], adapter=adapter), adapter=adapter):
        result = dspy.Predict(signature)(ticket="Payment failed.")
    for name, value in values.items():
        if rich or name == "severity":
            assert result[name].value == value.value
            assert result[name].confidence == pytest.approx(value.confidence)
            assert isinstance(result[name], decision_fields(True)[name])
            assert getattr(result[name], "probability", getattr(result[name], "probabilities", None)) is not None
        else:
            assert result[name] == value
            assert type(result[name]) is type(value)
    prepared = resolve_adapter(None, adapter, signature, {})._prepare(signature, [], {"ticket": "Payment failed."}, {})
    rendered = prepared["signature"]
    messages = adapter.format(rendered, [], prepared["inputs"])
    assert "Minor" in messages[0]["content"]
    schema = _get_structured_outputs_response_format(rendered).model_json_schema()
    if rich:
        assert "noul" in json.dumps(schema)
        assert "probabilities" in json.dumps(schema)
        assert '"value"' not in json.dumps(schema)
    else:
        assert schema["properties"]["urgent"]["type"] == "boolean"
        assert schema["properties"]["category"]["enum"] == ["billing", "technical"]


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("evidence", [False, True])
def test_inputs_to_both_backends_preserve_values_and_context(rich, evidence):
    signature = decision_signature(rich, "input")
    values = decision_values(rich, evidence)
    for adapter in (dspy.ChatAdapter(), dspy.JSONAdapter()):
        lm = DummyLM([{"accept": True}], adapter=adapter)
        with dspy.context(lm=lm, adapter=adapter):
            assert dspy.Predict(signature)(**values).accept is True
        messages = adapter.format(signature, [], values)
        assert "Minor" in messages[0]["content"]
        assert ("Product malfunction" in messages[0]["content"]) is rich
        prompt = messages[-1]["content"]
        assert '"confidence"' in prompt
        assert ('"probabilities"' in prompt) is evidence
        assert "1.5" in prompt
        assert "technical" in prompt
    client = FakeClient()
    module = dspy.Predict(signature)
    module.set_lm(client)
    module.fields["accept"] = {"threshold": 0.99}
    assert module(**values).accept is False
    state, questions = client.calls[0]
    assert set(questions) == {"accept"}  # Inputs are not additional questions.
    assert state["inputs"] == {
        name: v.model_dump(mode="json") if rich or name == "severity" else v for name, v in values.items()
    }
    assert "Minor" in state["input_fields"]
    assert values["urgent"].value is True if rich else values["urgent"] is True


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
@pytest.mark.parametrize("invalid", [-0.1, 2.1])
def test_score_range_is_enforced(adapter, invalid):
    sig = decision_signature(True, "output")
    values = decision_values(True)
    values["severity"] = {"value": invalid, "confidence": 0.7}
    completion = adapter.format_assistant_message_content(sig, values)
    with pytest.raises(dspy.AdapterParseError):
        adapter.parse(sig, completion)


@pytest.mark.parametrize(
    "kind,options",
    [
        (Score, ((0, "a"), (1, "b"))),  # Removed anchor syntax.
        (Score, ("a", 1)),
        (Score, ("a",)),
        (Score, "a"),
        (Choice, ((1, "a"), ("1", "b"))),
        (Choice, (([], "bad"),)),
    ],
)
def test_invalid_options(kind, options):
    with pytest.raises(ValueError):
        kind[options]


@pytest.mark.parametrize(
    "options",
    [
        ((1, "yes"), (False, "no")),
        ((True, "yes"), (0, "no")),
        ((True, "a"), (True, "b")),
        (("true", "yes"),),
        ((False, {"what": object()}),),
        (),
    ],
)
def test_noul_rejects_invalid_criteria(options):
    with pytest.raises(ValueError):
        Noul[options]


@pytest.mark.parametrize("kind", [Noul, Choice, Score])
def test_structured_description_defaults_are_detached_and_cacheable(kind):
    description = {"what": "Match", "examples": ["Original"], "metadata": {"enabled": True, "weight": 2}}

    def declare(desc):
        return kind[desc, []] if kind is Score else kind[(True, desc), (False, {})]

    annotation = declare(description)
    assert annotation is declare(description)
    if kind is not Score:
        assert kind[True, description] is kind[((True, description),)]
    sig = dspy.Signature({"result": (annotation, dspy.OutputField(desc="Assess."))})
    module = dspy.Predict(sig)
    expected = (
        [description, []]
        if kind is Score
        else {"true" if kind is Noul else "True": description, "false" if kind is Noul else "False": {}}
    )
    assert module.get_criteria("result") == expected
    snapshot = module.get_criteria("result")
    schema = annotation.model_json_schema()
    declared = annotation.criteria()
    entry = declared[0] if kind is Score else declared["true" if kind is Noul else "True"]
    entry["examples"].append("Type metadata mutation")
    assert declare(description).criteria() == snapshot
    assert annotation.model_json_schema() == schema
    description["examples"].append("Mutation")
    assert module.get_criteria("result") == snapshot
    returned = module.get_criteria("result")
    entry = returned[0] if kind is Score else returned["true" if kind is Noul else "True"]
    entry["examples"].append("Getter mutation")
    assert module.get_criteria("result") == snapshot
    assert module.fields == {}
    assert declare(description) is not annotation


@pytest.mark.parametrize("kind", [Noul, Choice, Score])
@pytest.mark.parametrize("description", [1, ("tuple",), {1: "bad key"}, {"bad": float("nan")}, {"bad": {1, 2}}])
def test_structured_descriptions_reject_non_json(kind, description):
    with pytest.raises(ValueError):
        if kind is Score:
            kind["low", description]
        else:
            kind[True, description]


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
def test_structured_criteria_reach_both_backends_and_survive_save_load(adapter, rich, tmp_path):
    yes = {"what": "Blocked", "examples": ["Login fails"]}
    no = {"what": "Usable"}
    technical = {"what": "Product bug", "not_for": "Billing"}
    urgent = Noul[(True, yes), (False, no)]
    category = Choice[("billing", {}), ("technical", technical)]
    severity = Score[None, ["Disruptive"], {"what": "Blocking"}]
    sig = dspy.Signature(
        {
            "ticket": (str, dspy.InputField()),
            "urgent": (urgent if rich else Annotated[bool, urgent], dspy.OutputField(desc="Is it blocked?")),
            "category": (
                category if rich else Annotated[Literal["billing", "technical"], category],
                dspy.OutputField(desc="Classify."),
            ),
            "severity": (severity, dspy.OutputField(desc="Rate impact.")),
        }
    )
    module = dspy.Predict(sig)
    expected = {
        "urgent": {"true": yes, "false": no},
        "category": {"billing": {}, "technical": technical},
        "severity": [None, ["Disruptive"], {"what": "Blocking"}],
    }
    evidence = {
        "urgent": {"noul": 0.8},
        "category": {"probabilities": {"billing": 0.1, "technical": 0.9}, "confidence": 0.7},
        "severity": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.6},
    }
    client = FakeClient(answers=evidence)
    jev = module(ticket="Login fails", lm=client)
    for name, criteria in expected.items():
        assert client.calls[-1][1][name]["criteria"] == criteria
    lm = DummyLM([evidence], adapter=adapter)
    with dspy.context(adapter=adapter):
        chat = module(ticket="Login fails", lm=lm)
    assert chat.toDict() == jev.toDict()
    assert (chat.urgent.value if rich else chat.urgent) is True
    assert (chat.category.value if rich else chat.category) == "technical"
    assert chat.severity.value == pytest.approx(1.6)
    translated = resolve_adapter(lm, adapter, sig, module.fields)._prepare(sig, [], {"ticket": "Login fails"}, {})
    for name, criteria in expected.items():
        assert json.loads(translated["signature"].output_fields[name].json_schema_extra["desc"])["criteria"] == criteria
    module.save(tmp_path / "defaults.json")
    restored = dspy.Predict(sig)
    restored.load(tmp_path / "defaults.json")
    assert restored.fields == {}
    for name, criteria in expected.items():
        assert restored.get_criteria(name) == criteria
    module.set_criteria("urgent", {"true": {"what": "Overridden"}})
    module.save(tmp_path / "overrides.json")
    restored.load(tmp_path / "overrides.json")
    assert restored.get_criteria("urgent") == {"true": {"what": "Overridden"}}
    assert dspy.Predict(sig).get_criteria("urgent") == expected["urgent"]


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
def test_noul_criteria_in_both_backends_inputs_and_outputs(rich, adapter):
    availability = Noul[(True, "Service unavailable"), (False, "Workaround available")]
    assert availability is Noul[(False, "Workaround available"), (True, "Service unavailable")]
    annotation = availability if rich else Annotated[bool, availability]
    output_sig = dspy.Signature(
        {"ticket": (str, dspy.InputField()), "unavailable": (annotation, dspy.OutputField(desc="Is service blocked?"))},
        "Assess operational availability.",
    )
    value = availability(value=False, confidence=0.8, probability=0.1) if rich else False
    with dspy.context(lm=DummyLM([{"unavailable": {"noul": 0.1}}], adapter=adapter), adapter=adapter):
        result = dspy.Predict(output_sig)(ticket="Use the workaround.").unavailable
    assert result == value
    assert type(result) is (availability if rich else bool)
    schema = _get_structured_outputs_response_format(output_sig).model_json_schema()
    assert ("Service unavailable" in json.dumps(schema)) is rich
    assert ("confidence" in json.dumps(schema)) is rich
    client = FakeClient(probability=0.3)
    result = dspy.Predict(output_sig)(ticket="Use the workaround.", lm=client).unavailable
    assert type(result) is (availability if rich else bool)
    assert (result.value if rich else result) is False
    assert client.calls[-1][1]["unavailable"] == {
        "type": "noul",
        "instructions": "Is service blocked?",
        "criteria": {"true": "Service unavailable", "false": "Workaround available"},
    }

    input_sig = dspy.Signature(
        {"unavailable": (annotation, dspy.InputField()), "accept": (bool, dspy.OutputField(desc="Accept the assessment?"))},
        "Read availability without reassessing it.",
    )
    with dspy.context(lm=DummyLM([{"accept": True}], adapter=adapter), adapter=adapter):
        assert dspy.Predict(input_sig)(unavailable=value).accept is True
    for sig, inputs in ((output_sig, {"ticket": "Use the workaround."}), (input_sig, {"unavailable": value})):
        prompt = adapter.format(sig, [], inputs)[0]["content"]
        assert "Service unavailable" in prompt
        assert "Workaround available" in prompt
    dspy.Predict(input_sig)(unavailable=value, lm=client)
    state = client.calls[-1][0]
    assert "Service unavailable" in state["input_fields"]
    assert state["inputs"]["unavailable"] == (value.model_dump(mode="json") if rich else False)


@pytest.mark.parametrize("levels,maximum", [(("Low", "High"), 1), (("Last", "Second", "Third", "First"), 3)])
def test_score_level_order_bounds_and_fractional_values(levels, maximum):
    score = Score[levels]
    assert score.criteria() == list(levels)
    assert json.dumps(list(enumerate(levels))) in score.description()
    adapter = TypeAdapter(score)
    value_schema = adapter.json_schema()["properties"]["value"]
    assert value_schema["minimum"] == 0
    assert value_schema["maximum"] == maximum
    for value in (0, 0.25, maximum):
        result = adapter.validate_python({"value": value, "confidence": 0.7})
        assert result.value == value
        assert float(result) == value
    with pytest.raises(ValidationError):
        adapter.validate_python({"value": maximum + 0.1, "confidence": 0.7})
    for level in (None, 0, maximum):
        result = score(value=0.25, confidence=0.7, level=level)
        assert score.model_validate_json(result.model_dump_json()).level == level
    for level in (-1, maximum + 1, 0.5, True):
        data = {"value": 0.25, "confidence": 0.7, "level": level}
        with pytest.raises(ValidationError):
            score(**data)
        with pytest.raises(ValidationError):
            score.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("score", [Score, Severity])
@pytest.mark.parametrize("field", [dspy.InputField, dspy.OutputField])
def test_score_cannot_annotate_a_native_float(score, field):
    with pytest.raises(ValueError, match=r"Use Score.*directly"):
        dspy.Signature({"rating": (Annotated[float, score], field())})


def test_bare_float_remains_an_ordinary_llm_output():
    with dspy.context(lm=DummyLM([{"rating": 12.75}])):
        result = dspy.Predict("text -> rating: float")(text="x")
    assert type(result.rating) is float
    assert result.rating == 12.75


def test_choice_type_cache_preserves_bool_vs_int():
    boolean = Choice[(True, "yes"), (False, "no")]
    integer = Choice[(1, "yes"), (0, "no")]
    assert boolean is not integer
    assert type(integer(value=1, confidence=0.5).value) is int
    assert type(boolean(value=True, confidence=0.5).value) is bool


@pytest.mark.parametrize("kind,value", [(Noul, True), (Severity, 1.2), (Category, "technical")])
def test_rich_confidence_required_and_serialization(kind, value):
    with pytest.raises(ValidationError):
        kind(value=value)
    with pytest.raises(ValidationError):
        kind(value=value, confidence=1.1)
    result = kind(value=value, confidence=0.7)
    assert TypeAdapter(kind).validate_json(result.model_dump_json()) == result
    assert result.model_dump(mode="json") == {"value": value, "confidence": 0.7}


def test_jev_result_can_feed_llm_and_back():
    source = dspy.Predict(decision_signature(True, "output"))
    source.set_lm(FakeClient(choice="technical"))
    result = source(ticket="Payment failed.")
    adapter = dspy.ChatAdapter()
    with dspy.context(lm=DummyLM([decision_evidence(True)], adapter=adapter), adapter=adapter):
        regenerated = dspy.Predict(decision_signature(True, "output"))(ticket="Payment failed.")
    target = dspy.Predict(decision_signature(True, "input"))
    target.set_lm(FakeClient())
    assert target(**dict(result.items())).accept is True
    assert target(**dict(regenerated.items())).accept is True
    with dspy.context(lm=DummyLM([{"accept": True}])):
        assert dspy.Predict(decision_signature(True, "input"))(**dict(result.items())).accept is True


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
@pytest.mark.parametrize("selected", [1, True, None, "雪"])
@pytest.mark.asyncio
async def test_annotated_literal_evidence_criteria_and_native_types(adapter, selected, tmp_path):
    choice = Choice[(1, "integer"), (True, "boolean"), (None, "missing"), ("雪", "snow")]
    annotation = Annotated[Literal[1, True, None, "雪"], choice]
    field = TypeAdapter(annotation)
    assert type(field.validate_python(selected)) is type(selected)
    with pytest.raises(ValidationError):
        field.validate_python("unknown")
    schema = field.json_schema()
    assert "confidence" not in json.dumps(schema)
    assert "snow" in schema["description"]
    sig = dspy.Signature({"text": (str, dspy.InputField()), "label": (annotation, dspy.OutputField(desc="Choose a label."))})
    probabilities = {label: 0.7 if label == str(selected) else 0.1 for label in choice.criteria()}
    evidence = {"label": {"probabilities": probabilities, "confidence": 0.6}}
    lm = DummyLM([evidence], adapter=adapter)
    client = FakeClient(answers=evidence)
    module = dspy.Predict(sig)
    with dspy.context(adapter=adapter):
        a, b = module(text="x", lm=lm).label, module(text="x", lm=client).label
        assert type((await module.acall(text="x", lm=client)).label) is type(selected)
    assert a == b == selected
    assert type(a) is type(b) is type(selected)
    assert client.calls[0][1]["label"]["criteria"] == choice.criteria()
    module.save(tmp_path / "annotated.json")
    restored = dspy.Predict(sig)
    restored.load(tmp_path / "annotated.json")
    assert type(restored(text="x", lm=client).label) is type(selected)
    input_sig = dspy.Signature({"label": (annotation, dspy.InputField()),
                                "accept": (bool, dspy.OutputField(desc="Accept the label?"))})
    inputs = FakeClient()
    dspy.Predict(input_sig)(label=selected, lm=inputs)
    assert type(inputs.calls[0][0]["inputs"]["label"]) is type(selected)
    assert "snow" in inputs.calls[0][0]["input_fields"]
    assert "snow" in adapter.format(input_sig, [], {"label": selected})[0]["content"]


@pytest.mark.parametrize("literal", [Literal[1], Literal[True, "extra"], Literal["True"]])
@pytest.mark.parametrize("field", [dspy.InputField, dspy.OutputField])
def test_annotated_literal_rejects_mismatched_criteria(literal, field):
    with pytest.raises(ValueError, match="Choice criteria must match"):
        dspy.Signature({"label": (Annotated[literal, Choice[(True, "yes")]], field())})


def test_bare_choice_metadata_uses_literal_members():
    annotation = Annotated[Literal["a", "b"], Choice]
    sig = dspy.Signature({"label": (annotation, dspy.OutputField(desc="Choose a label."))})
    client = FakeClient(choice="b")
    assert dspy.Predict(sig)(lm=client).label == "b"
    assert client.calls[0][1]["label"]["criteria"] == {"a": None, "b": None}


def test_annotated_choice_uses_criteria_order_and_saved_weights(tmp_path):
    annotation = Annotated[Literal["a", "b"], Choice[("b", "second"), ("a", "first")]]
    sig = dspy.Signature({"label": (annotation, dspy.OutputField(desc="Choose a label."))})
    client = FakeClient(answers={"label": {"probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.4}})
    module = dspy.Predict(sig)
    assert module(lm=client).label == "b"
    module.fields["label"] = {"weights": {"b": 0}}
    module.save(tmp_path / "weights.json")
    restored = dspy.Predict(sig)
    restored.load(tmp_path / "weights.json")
    assert restored(lm=client).label == "a"
