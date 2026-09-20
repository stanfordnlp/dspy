import json
from typing import Annotated, Literal

import pytest
from pydantic import TypeAdapter, ValidationError

import dspy
from dspy.adapters.json_adapter import _get_structured_outputs_response_format
from dspy.utils.dummies import DummyLM
from tests.predict.test_decide import FakeClient

Severity = dspy.Score[(0, "Minor"), (2, "Disruptive"), (10, "Blocking")]
Category = dspy.Choice[("billing", "Payment issue"), ("technical", "Product malfunction")]


def decision_fields(rich):
    return {
        "urgent": dspy.Noul if rich else bool,
        "severity": Severity if rich else Annotated[float, Severity],
        "category": Category if rich else Literal["billing", "technical"],
    }


def decision_signature(rich, direction):
    fields = {
        name: (kind, dspy.InputField() if direction == "input" else dspy.OutputField())
        for name, kind in decision_fields(rich).items()
    }
    if direction == "input":
        fields["accept"] = (bool, dspy.OutputField())
    else:
        fields = {"ticket": (str, dspy.InputField()), **fields}
    return dspy.Signature(fields, "Assess the ticket.")


def decision_values(rich, evidence=False):
    if not rich:
        return {"urgent": True, "severity": 6.6, "category": "technical"}
    return {
        "urgent": dspy.Noul(value=True, confidence=0.6, **({"probability": 0.8} if evidence else {})),
        "severity": Severity(
            value=6.6, confidence=0.61, **({"probabilities": {0: 0.1, 1: 0.3, 2: 0.6}} if evidence else {})
        ),
        "category": Category(
            value="technical",
            confidence=0.73,
            **({"probabilities": {"billing": 0.2, "technical": 0.8}} if evidence else {}),
        ),
    }


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
@pytest.mark.parametrize("rich", [False, True])
def test_predict_outputs_and_generated_schema(adapter, rich):
    signature = decision_signature(rich, "output")
    values = decision_values(rich)
    with dspy.context(lm=DummyLM([values], adapter=adapter), adapter=adapter):
        result = dspy.Predict(signature)(ticket="Payment failed.")
    for name, value in values.items():
        assert result[name] == value
        if rich:
            assert isinstance(result[name], decision_fields(True)[name])
            assert getattr(result[name], "probability", getattr(result[name], "probabilities", None)) is None
        else:
            assert type(result[name]) is type(value)
    messages = adapter.format(signature, [], {"ticket": "Payment failed."})
    assert "Minor" in messages[0]["content"]
    schema = _get_structured_outputs_response_format(signature).model_json_schema()
    if rich:
        for definition in schema["$defs"].values():
            assert set(definition["properties"]) == {"value", "confidence"}
            assert set(definition["required"]) == {"value", "confidence"}
    else:
        assert "confidence" not in json.dumps(schema)
        assert schema["properties"]["severity"]["minimum"] == 0
        assert schema["properties"]["severity"]["maximum"] == 10
        assert schema["properties"]["category"]["enum"] == ["billing", "technical"]


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("evidence", [False, True])
def test_inputs_to_both_modules_preserve_values_and_context(rich, evidence):
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
        assert ('"confidence"' in prompt) is rich
        assert ('"probabilities"' in prompt) is (rich and evidence)
        assert "6.6" in prompt
        assert "technical" in prompt
    client = FakeClient()
    module = dspy.Decide(signature, client=client)
    module.thresholds["accept"] = 0.99
    assert module(**values).accept is False
    state, questions = client.calls[0]
    assert set(questions) == {"accept"}  # Inputs are not additional questions.
    assert state == {name: v.model_dump(mode="json") if rich else v for name, v in values.items()}
    assert "Minor" in questions["accept"]["instructions"]["inputs"]
    assert values["urgent"].value is True if rich else values["urgent"] is True


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
def test_score_range_is_enforced_for_native_and_rich(adapter, rich):
    sig = decision_signature(rich, "output")
    values = decision_values(False)
    values["severity"] = 11
    if rich:
        values = {name: {"value": value, "confidence": 0.7} for name, value in values.items()}
    completion = adapter.format_assistant_message_content(sig, values)
    with pytest.raises(dspy.AdapterParseError):
        adapter.parse(sig, completion)


@pytest.mark.parametrize(
    "kind,options",
    [
        (dspy.Score, ((2, "a"), (1, "b"))),
        (dspy.Score, ((1, "a"), (1, "b"))),
        (dspy.Score, ((0, "a"), (float("nan"), "b"))),
        (dspy.Score, ((0, "a"),)),
        (dspy.Choice, ((1, "a"), ("1", "b"))),
        (dspy.Choice, (([], "bad"),)),
    ],
)
def test_invalid_options(kind, options):
    with pytest.raises(ValueError):
        kind[options]


def test_choice_type_cache_preserves_bool_vs_int():
    boolean = dspy.Choice[(True, "yes"), (False, "no")]
    integer = dspy.Choice[(1, "yes"), (0, "no")]
    assert boolean is not integer
    assert type(integer(value=1, confidence=0.5).value) is int
    assert type(boolean(value=True, confidence=0.5).value) is bool


@pytest.mark.parametrize("kind,value", [(dspy.Noul, True), (Severity, 4.2), (Category, "technical")])
def test_rich_confidence_required_and_serialization(kind, value):
    with pytest.raises(ValidationError):
        kind(value=value)
    with pytest.raises(ValidationError):
        kind(value=value, confidence=1.1)
    result = kind(value=value, confidence=0.7)
    assert TypeAdapter(kind).validate_json(result.model_dump_json()) == result
    assert result.model_dump(mode="json") == {"value": value, "confidence": 0.7}


def test_decide_result_can_feed_predict_and_back():
    source = dspy.Decide(decision_signature(True, "output"), client=FakeClient(choice="technical"))
    result = source(ticket="Payment failed.")
    adapter = dspy.ChatAdapter()
    with dspy.context(lm=DummyLM([decision_values(True)], adapter=adapter), adapter=adapter):
        regenerated = dspy.Predict(decision_signature(True, "output"))(ticket="Payment failed.")
    target = dspy.Decide(decision_signature(True, "input"), client=FakeClient())
    assert target(**dict(result.items())).accept is True
    assert target(**dict(regenerated.items())).accept is True
    with dspy.context(lm=DummyLM([{"accept": True}])):
        assert dspy.Predict(decision_signature(True, "input"))(**dict(result.items())).accept is True
