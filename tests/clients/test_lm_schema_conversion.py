"""Canonical requests preserve strict preparation of generated Pydantic schemas."""

import copy
import json

import pydantic
import pytest

import dspy
from dspy.clients.execution import prepare
from dspy.clients.requests import build_request, close_object_schemas


@pytest.mark.parametrize("asynchronous", [False, True])
def test_generated_nested_schema_is_prepared_without_mutation(asynchronous):
    class Item(pydantic.BaseModel):
        name: str

    class Answer(pydantic.BaseModel):
        items: list[Item]
        alternative: Item | str

    original = copy.deepcopy(Answer.model_json_schema())
    lm = dspy.LM("openai/gpt-4.1-mini", model_type="responses", engine="lm15", cache=False)
    call = prepare(lm, "Return an answer.", {"response_format": Answer}, asynchronous=asynchronous)
    schema = call.request.config.response_format["schema"]

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Item"]["additionalProperties"] is False
    assert call.request.config.response_format["strict"] is True
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
    call = prepare(lm, "Return an answer.", {"response_format": format_})

    assert call.request.config.response_format["schema"] == original["json_schema"]["schema"]
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
    lm = dspy.LM("openai/gpt-4o", engine="lm15", cache=False)
    schema = build_request(lm, "hi", {"response_format": _Record}).config.response_format["schema"]
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
    out = close_object_schemas(schema)
    assert out["properties"]["a"] == {"type": "string", "description": "d"}
    assert out["properties"]["b"] == {"$ref": "#/$defs/A"}  # a bare ref stays a ref
    assert out["required"] == ["a", "b"]


@pytest.mark.parametrize("engine", ["lm15", "litellm"])
def test_pydantic_response_format_is_strict_on_both_engines(engine):
    lm = dspy.LM("openai/gpt-4o", engine=engine, api_key="k", cache=False)
    request = prepare(lm, "hi", {"response_format": _Record}).request
    fmt = request.config.response_format
    assert fmt["type"] == "json_schema" and fmt.get("strict") is True
    assert fmt["schema"]["required"] == ["name", "count", "tag", "inner", "items"]
    assert "default" not in fmt["schema"]["properties"]["tag"]


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_function_tool_strictness_is_not_silently_weakened(flat, strict):
    from dspy.utils.exceptions import LMUnsupportedFeatureError

    function = {"name": "lookup", "parameters": {"type": "object", "properties": {}}, "strict": strict}
    tool = {"type": "function", **function} if flat else {"type": "function", "function": function}
    original = copy.deepcopy(tool)
    lm = dspy.LM("openai/gpt-4.1-mini", model_type="responses", engine="lm15")
    if strict:
        with pytest.raises(LMUnsupportedFeatureError, match="strict"):
            build_request(lm, "Find a record.", {"tools": [tool]})
    else:
        request = build_request(lm, "Find a record.", {"tools": [tool]})
        assert request.tools[0].name == "lookup"
        assert request.tools[0].parameters == function["parameters"]
    assert tool == original


@pytest.mark.parametrize("reasoning", [{"enabled": False}, {"effort": "none"}, {"effort": "off"}])
@pytest.mark.parametrize("model_type", ["chat", "responses"])
def test_reasoning_disable_survives_request_serialization(reasoning, model_type):
    from dspy.clients.lm15_boundary import request_kwargs

    lm = dspy.LM("openai/gpt-5", model_type=model_type, engine="lm15", reasoning_effort="high")
    request = build_request(lm, "Answer briefly.", {"reasoning": reasoning})
    assert request.config.reasoning.effort == "off"
    assert request.config.reasoning.summary is None
    wire = request_kwargs(request, model_type)
    if model_type == "responses":
        assert wire["reasoning"] == {"effort": "none"}
    else:
        assert wire["reasoning_effort"] == "none"


def test_enabled_reasoning_retains_effort_and_summary():
    lm = dspy.LM("openai/gpt-5", engine="lm15")
    request = build_request(lm, "Explain.", {"reasoning": {"effort": "low", "summary": "concise"}})
    assert request.config.reasoning.effort == "low"
    assert request.config.reasoning.summary == "concise"


@pytest.mark.parametrize("option", [{"reasoning_effort": "high"}, {"reasoning": {"effort": "high"}}])
def test_native_reasoning_controls_keep_their_combined_validation(option):
    lm = dspy.LM("deepseek/deepseek-chat", engine="auto")
    request = build_request(lm, "Explain.", {**option, "thinking": {"type": "enabled"}})
    assert request.config.reasoning.effort == "high"
    with pytest.raises(ValueError, match="contradictory"):
        build_request(lm, "Explain.", {**option, "thinking": {"type": "disabled"}})


@pytest.mark.parametrize("reasoning", [{"effort": "high"}, {"enabled": False}, {"effort": "none"}])
def test_reasoning_dictionary_respects_native_disable(reasoning):
    lm = dspy.LM("deepseek/deepseek-chat", engine="auto")
    options = {"reasoning": reasoning, "thinking": {"type": "disabled"}}
    if reasoning.get("effort") == "high":
        with pytest.raises(ValueError, match="contradictory"):
            build_request(lm, "Explain.", options)
    else:
        assert build_request(lm, "Explain.", options).config.reasoning.is_off


@pytest.mark.parametrize("defaults", [{"reasoning": {"effort": "high"}}, {"reasoning_effort": "high"}])
@pytest.mark.parametrize("name", ["reasoning", "reasoning_effort"])
@pytest.mark.parametrize("clear", [False, True])
def test_reasoning_aliases_override_or_clear_defaults(defaults, name, clear):
    from dspy.clients.lm15_boundary import request_kwargs

    lm = dspy.LM("openai/gpt-5", engine="lm15", **defaults)
    value = None if clear else {"effort": "low"} if name == "reasoning" else "low"
    request = build_request(lm, "Explain.", {name: value})
    if clear:
        assert request.config.reasoning is None
        assert "reasoning" not in request_kwargs(request, "responses")
    else:
        assert request.config.reasoning.effort == "low"
        assert request_kwargs(request, "responses")["reasoning"]["effort"] == "low"
    assert build_request(lm, "Explain.").config.reasoning.effort == "high"


@pytest.mark.parametrize("model", ["openai/gpt-5", "openai/gpt-4o"])
@pytest.mark.parametrize("name", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", [32000, None])
def test_token_limit_aliases_override_or_clear_defaults(model, name, value):
    lm = dspy.LM(model, engine="lm15", max_tokens=16000)
    assert build_request(lm, "hello", {name: value}).config.max_tokens == value
    assert build_request(lm, "hello").config.max_tokens == 16000


def test_conflicting_token_limits_in_one_call_are_still_rejected():
    lm = dspy.LM("openai/gpt-4o", engine="lm15")
    with pytest.raises(ValueError, match="disagree"):
        build_request(lm, "hello", {"max_tokens": 100, "max_completion_tokens": 200})


@pytest.mark.parametrize("model,options,field,expected", [
    ("openrouter/openai/gpt-5", {"reasoning_effort": "low"}, "reasoning", {"effort": "low"}),
    ("openrouter/openai/gpt-5", {"reasoning_effort": "none"}, "reasoning", {"enabled": False}),
    ("groq/compound", {"tools": [{"type": "browser_search"}]}, "tools", [{"type": "browser_search"}]),
    ("groq/compound", {"tools": [{"type": "code_interpreter"}]}, "tools", [{"type": "code_interpreter"}]),
])
def test_provider_option_spellings_survive_native_serialization(model, options, field, expected):
    from dspy._vendor.lm15.testing import FakeResponse, FakeTransport
    from dspy.clients.engines import LM15Engine
    from dspy.lm15 import RouterConfig

    lm = dspy.LM(model, engine="auto")
    request = build_request(lm, "hello", options)
    raw = {"id": "reply", "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]}
    transport = FakeTransport([FakeResponse(status=200, body=json.dumps(raw).encode())])
    engine = LM15Engine(RouterConfig(api_keys={"openrouter": "test", "groq": "test"}, transport=transport))
    assert engine.complete(request).text == "hello"
    assert json.loads(transport.requests[0].body)[field] == expected


@pytest.mark.parametrize("route", [{"engine": "litellm"}, {"engine": "auto", "extra_headers": {"X-Gateway": "test"}}])
@pytest.mark.parametrize("control", [
    {"thinking": {"type": "disabled"}},
    {"thinking": {"type": "enabled", "budget_tokens": 2048}},
    {"enable_thinking": False},
    {"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}},
    {"reasoning_format": "raw"},
])
def test_compatibility_routes_preserve_provider_thinking_controls(route, control, monkeypatch):
    import litellm

    seen = []

    def complete(**kwargs):
        seen.append(kwargs)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "hello"},
                                               "finish_reason": "stop"}])

    monkeypatch.setattr(litellm, "completion", complete)
    lm = dspy.LM("deepseek/deepseek-chat", cache=False, num_retries=0, **route)
    request = build_request(lm, "hello", control)
    assert lm(request).text == "hello"
    for key, value in control.items():
        assert seen[0][key] == value
    assert "reasoning_effort" not in seen[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("engine", ["auto", "lm15"])
@pytest.mark.parametrize("thinking", [{"type": "enabled", "budget_tokens": 2048}, {"type": "disabled"}])
async def test_native_anthropic_preserves_provider_thinking_controls(asynchronous, engine, thinking, monkeypatch):
    import dspy.clients.execution as execution
    from dspy._vendor.lm15.testing import FakeTransport
    from dspy.lm15 import RouterConfig
    from tests.clients.test_lm_engine_execution import DualResponse

    raw = {"id": "reply", "model": "claude-sonnet-4-5", "role": "assistant", "type": "message",
           "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
           "usage": {"input_tokens": 3, "output_tokens": 1}}
    transport = FakeTransport([DualResponse(status=200, body=json.dumps(raw).encode())])
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**kwargs, "api_keys": {"anthropic": "test"}, "transport": transport},
    ))
    lm = dspy.LM("anthropic/claude-sonnet-4-5", engine=engine, cache=False, num_retries=0, max_tokens=4096)
    request = build_request(lm, "hello", {"thinking": thinking})
    result = await lm.acall(request) if asynchronous else lm(request)
    assert result.text == "hello"
    body = json.loads(transport.requests[0].body)
    assert body["thinking"] == thinking
    assert body["max_tokens"] == 4096
    await lm.aclose()


@pytest.mark.parametrize("thinking", [
    {"type": "enabled", "budget_tokens": 2048}, {"type": "adaptive"}, {"type": "disabled"},
])
@pytest.mark.parametrize("option,off", [
    ({"reasoning_effort": "none"}, True), ({"reasoning": {"enabled": False}}, True),
    ({"reasoning_effort": "high"}, False),
])
def test_native_anthropic_validates_thinking_defaults_against_reasoning_overrides(thinking, option, off):
    lm = dspy.LM("anthropic/claude-sonnet-4-5", engine="auto", thinking=thinking)
    if off != (thinking["type"] == "disabled"):
        with pytest.raises(ValueError, match="contradictory"):
            build_request(lm, "hello", option)
    else:
        request = build_request(lm, "hello", option)
        assert request.config.reasoning.is_off == off
        assert request.config.extensions["thinking"] == thinking
