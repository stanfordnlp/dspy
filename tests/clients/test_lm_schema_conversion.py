"""Generated Pydantic schemas receive the same preparation on both LM paths."""

import copy

import pydantic
import pytest

import dspy
from dspy.clients.execution import _canonical, prepare
from dspy.clients.legacy_requests import chat_to_responses


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
