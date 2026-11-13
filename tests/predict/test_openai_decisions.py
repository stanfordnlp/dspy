import copy
import json
from typing import Literal

import pytest

import dspy
from dspy.adapters.decision import record_evidence
from dspy.experimental import Choice, Noul, Score
from dspy.utils.exceptions import AdapterParseError, LMAuthError, LMUnsupportedFeatureError
from tests.clients.test_lm_engine_execution import DualResponse, native_transport

Department = Choice[("billing", "Invoices and payments"), ("technical", "Service incidents")]
Severity = Score["no impact", "partial outage", "total outage"]


class Triage(dspy.Signature):
    """Route the ticket using the supplied evidence."""

    ticket: str = dspy.InputField()
    urgent: Noul = dspy.OutputField(desc="Is service completely down?")
    department: Department = dspy.OutputField(desc="Which team should respond?")
    severity: Severity = dspy.OutputField(desc="Rate service impact.")


BODY = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "urgent", "probability": 0.7},
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": [{"value": "billing", "probability": 0.8}, {"value": "technical", "probability": 0.2}],
        },
        {
            "type": "score",
            "name": "severity",
            "score": 1.5,
            "confidence": 0.6,
            "probabilities": [
                {"value": 0, "label": "0", "probability": 0.1},
                {"value": 1, "label": "1", "probability": 0.3},
                {"value": 2, "label": "2", "probability": 0.6},
            ],
        },
    ],
    "usage": {
        "input_tokens": 123,
        "output_tokens": 0,
        "total_tokens": 123,
        "input_tokens_details": {"cached_tokens": 12, "cache_write_tokens": 10},
    },
}


def program(monkeypatch, body=BODY, status=200, **kwargs):
    wire = native_transport(monkeypatch, [DualResponse(status=status, body=json.dumps(body).encode())])
    lm = dspy.LM("openai-decisions/gpt-6-luna", api_key="test-key", **kwargs)
    predict = dspy.Predict(Triage)
    predict.set_lm(lm)
    predict.fields = {
        "urgent": {"threshold": 0.8},
        "department": {"weights": {"technical": 5}},
        "severity": {"cuts": [0.5, 1.6]},
    }
    return predict, wire


def assert_prediction(result):
    assert result.urgent.value is False
    assert result.urgent.probability == 0.7
    assert result.department.value == "technical"
    assert result.department.probabilities == {"billing": 0.8, "technical": 0.2}
    assert result.severity.value == pytest.approx(1.5)
    assert result.severity.level == 1


def test_predict_thresholds_weights_evidence_usage_cache_and_demos(monkeypatch):
    predict, wire = program(monkeypatch)
    predict.demos = [
        dspy.Example(ticket="example", urgent=False, department="billing", severity=0).with_inputs("ticket")
    ]
    with dspy.context(track_usage=True), record_evidence() as evidence:
        result = predict(ticket="Charged twice.")
    assert_prediction(result)
    assert len(evidence) == 3
    assert result.get_lm_usage()[predict.lm.model]["total_tokens"] == 123
    assert predict.lm.history[-1]["usage"]["prompt_tokens"] == 123
    assert_prediction(predict(ticket="Charged twice."))
    assert len(wire.requests) == 1
    assert wire.requests[0].url == "https://api.openai.com/v1/decisions"
    payload = json.loads(wire.requests[0].body)
    assert payload["model"] == "gpt-6-luna"
    state = json.loads(payload["input"][0]["content"][0]["text"])
    assert state["inputs"] == {"ticket": "Charged twice."}
    assert state["demos"][0]["ticket"] == "example"
    assert [q["type"] for q in payload["questions"]] == ["predicate", "choice", "score"]
    assert not predict.lm.supports_reasoning
    assert not predict.lm.supports_function_calling
    assert predict.lm.supported_params == {"response_format"}


@pytest.mark.asyncio
async def test_async_predict(monkeypatch):
    predict, wire = program(monkeypatch)
    try:
        assert_prediction(await predict.acall(ticket="Charged twice."))
        assert wire.requests[0].url.endswith("/decisions")
    finally:
        await predict.lm.aclose()


def test_save_load_keeps_provider_and_parameters(monkeypatch, tmp_path):
    predict, _ = program(monkeypatch)
    path = tmp_path / "triage.json"
    predict.save(path)
    assert "test-key" not in path.read_text()
    restored = dspy.Predict(Triage)
    restored.load(path)
    assert restored.lm.model == predict.lm.model
    assert restored.lm.supports_decision_requests
    assert restored.fields == predict.fields


def test_native_bool_and_literal_outputs(monkeypatch):
    predict, _ = program(monkeypatch)
    signature = Triage.with_updated_fields("urgent", type_=bool).with_updated_fields(
        "department", type_=Literal["billing", "technical"]
    )
    predict = dspy.Predict(signature)
    predict.set_lm(dspy.LM("openai-decisions:gpt-6-luna", api_key="test-key"))
    result = predict(ticket="Charged twice.")
    assert result.urgent is True
    assert result.department == "billing"


def test_refusal_is_not_a_negative_label_and_usage_is_retained(monkeypatch):
    body = copy.deepcopy(BODY)
    body["answers"][0] = {"name": "urgent", "type": "refusal"}
    predict, wire = program(monkeypatch, body)
    with pytest.raises(AdapterParseError, match="refused outputs"):
        predict(ticket="Refused input")
    assert len(wire.requests) == 1
    assert predict.lm.history[-1]["usage"]["prompt_tokens"] == 123


def test_auth_error_uses_existing_dspy_boundary(monkeypatch):
    predict, wire = program(
        monkeypatch,
        {"error": {"type": "invalid_request_error", "code": "invalid_api_key", "message": "Incorrect API key"}},
        status=401,
    )
    with pytest.raises(LMAuthError):
        predict(ticket="anything")
    assert len(wire.requests) == 1


def test_unsupported_output_and_options_do_not_fall_back(monkeypatch):
    predict, wire = program(monkeypatch)
    box = dspy.Predict(
        dspy.Signature({"ticket": (str, dspy.InputField()), "box": (dspy.BoundingBox, dspy.OutputField(desc="Locate"))})
    )
    box.set_lm(predict.lm)
    with pytest.raises(LMUnsupportedFeatureError, match="bounding boxes"):
        box(ticket="image")
    with pytest.raises(ValueError, match="generation settings"):
        predict(ticket="anything", config={"temperature": 0.2})
    with pytest.raises(LMUnsupportedFeatureError):
        predict.lm("free-form chat is unsupported")
    assert not wire.requests


def test_lm_generation_defaults_are_not_silently_discarded(monkeypatch):
    predict, wire = program(monkeypatch, temperature=0.2)
    with pytest.raises(ValueError, match="generation settings"):
        predict(ticket="anything")
    assert not wire.requests


@pytest.mark.parametrize(
    "kwargs",
    [
        {"engine": "litellm"},
        {"engine": "litellm", "model_type": "chat"},
        {"model_type": "responses"},
        {"model_type": "text"},
        {"extra_headers": {"x-test": "1"}},
    ],
)
def test_unsupported_backend_configuration_is_not_silently_rerouted(monkeypatch, kwargs):
    predict, wire = program(monkeypatch, **kwargs)
    with pytest.raises(LMUnsupportedFeatureError, match="no LiteLLM or chat fallback"):
        predict(ticket="anything")
    assert not wire.requests
