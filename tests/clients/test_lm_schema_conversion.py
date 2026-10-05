"""Generated Pydantic schemas receive the same preparation on both LM paths."""

import copy

import pydantic
import pytest

import dspy
from dspy.clients.execution import _canonical, prepare
from dspy.clients.legacy_requests import _strict_json_schema, chat_to_responses


@pytest.mark.parametrize("asynchronous", [False, True])
def test_generated_nested_schema_matches_legacy_preparation(asynchronous):
    class Item(pydantic.BaseModel):
        name: str

    class Answer(pydantic.BaseModel):
        items: list[Item]
        alternative: Item | str

    original = copy.deepcopy(Answer.model_json_schema())
    lm = dspy.LM("openai/gpt-4.1-mini", model_type="responses", engine="lm15", cache=False)
    call = prepare(lm, "Return an answer.", None, {"response_format": Answer}, asynchronous=asynchronous)
    canonical = _canonical(call)
    schema = canonical.config.response_format["schema"]
    legacy = chat_to_responses(call.legacy)["text"]["format"]["schema"]

    assert schema == legacy
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Item"]["additionalProperties"] is False
    assert canonical.config.response_format["strict"] is True
    assert Answer.model_json_schema() == original
    assert "additionalProperties" not in original


@pytest.mark.parametrize("additional_properties", [None, True, False])
def test_raw_schema_is_not_rewritten(additional_properties):
    schema = {"type": "object", "properties": {"word": {"type": "string"}}}
    if additional_properties is not None:
        schema["additionalProperties"] = additional_properties
    format_ = {"type": "json_schema", "json_schema": {"name": "Answer", "schema": schema, "strict": True}}
    original = copy.deepcopy(format_)
    lm = dspy.LM("openai/gpt-4.1-mini", model_type="responses", engine="lm15", cache=False)
    call = prepare(lm, "Return an answer.", None, {"response_format": format_})

    assert _canonical(call).config.response_format["schema"] == original["json_schema"]["schema"]
    assert format_ == original


# ─── a generated response_format schema is shaped for OpenAI strict mode ──


class _Inner(pydantic.BaseModel):
    note: str = "n/a"


class _Record(pydantic.BaseModel):
    name: str
    count: int = 0
    tag: str | None = None
    inner: _Inner | None = None
    items: list[_Inner] = []


def test_strict_schema_requires_every_property_and_drops_none_defaults():
    # OpenAI strict mode rejected any pydantic model with defaults or an
    # optional submodel ("required must include every key"). The shape now
    # follows the OpenAI SDK's own to_strict_json_schema rules.
    schema = _strict_json_schema(_Record.model_json_schema())
    assert schema["required"] == ["name", "count", "tag", "inner", "items"]
    assert schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["tag"]  # None default dropped
    assert schema["properties"]["count"]["default"] == 0  # a value default is kept, as the OpenAI SDK keeps it
    inner = schema["$defs"]["_Inner"]
    assert inner["required"] == ["note"] and inner["additionalProperties"] is False
    assert {"$ref": "#/$defs/_Inner"} in schema["properties"]["inner"]["anyOf"]


def test_strict_schema_unravels_a_ref_with_siblings_and_a_lone_allof():
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/A", "description": "d"},
                                               "b": {"allOf": [{"$ref": "#/$defs/A"}]}},
              "$defs": {"A": {"type": "string"}}}
    out = _strict_json_schema(schema)
    assert out["properties"]["a"] == {"type": "string", "description": "d"}
    assert out["properties"]["b"] == {"$ref": "#/$defs/A"}  # a bare ref stays a ref
    assert out["required"] == ["a", "b"]


def test_pydantic_response_format_is_sent_strict_on_both_paths():
    lm = dspy.LM("openai/gpt-4o", engine="lm15", api_key="k", cache=False)
    request = _canonical(prepare(lm, "hi", None, {"response_format": _Record}))
    fmt = request.config.response_format
    assert fmt["type"] == "json_schema" and fmt.get("strict") is True
    assert fmt["schema"]["required"] == ["name", "count", "tag", "inner", "items"]
    # The forced LiteLLM Responses path converts the legacy body itself and
    # must mean the same contract.
    data = chat_to_responses({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}],
                              "response_format": _Record})
    fmt = data["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"]["required"] == ["name", "count", "tag", "inner", "items"]
    assert "default" not in fmt["schema"]["properties"]["tag"]
