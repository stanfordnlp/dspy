"""Explicitly selected live coverage for canonical requests and ordinary calls."""

import json
import os

import pydantic
import pytest

import dspy
from dspy.lm15 import Config, Message, Request, Response, ToolCallPart


def _require_env(*keys):
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing live LM credentials: {', '.join(missing)}")


def _text(response):
    assert isinstance(response, Response)
    assert response.text is not None
    return response.text.strip()


@pytest.mark.llm_call
@pytest.mark.parametrize("model,model_type,key", [
    ("openai/gpt-4.1-mini", "chat", "OPENAI_API_KEY"),
    ("openai/gpt-4.1-mini", "responses", "OPENAI_API_KEY"),
    ("anthropic/claude-haiku-4-5", "chat", "ANTHROPIC_API_KEY"),
    ("gemini/gemini-2.5-flash", "chat", "GEMINI_API_KEY"),
])
def test_live_canonical_multiturn_and_response_replay(model, model_type, key):
    _require_env(key)
    lm = dspy.LM(model, model_type=model_type, engine="lm15", cache=False, num_retries=0)
    request = Request(model=model, system="Answer briefly.", messages=(Message.user("Say DSPy."),),
                      config=Config(max_tokens=256))
    try:
        first = lm(request)
        followup = Request(model=model, system=request.system,
                           messages=(*request.messages, first.message, Message.user("Repeat your answer.")),
                           config=request.config)
        assert "dspy" in _text(lm(followup)).lower()
    finally:
        lm.close()


@pytest.mark.llm_call
@pytest.mark.parametrize("model,model_type,key", [
    ("openai/gpt-4.1-mini", "chat", "OPENAI_API_KEY"),
    ("openai/gpt-4.1-mini", "responses", "OPENAI_API_KEY"),
    ("anthropic/claude-haiku-4-5", "chat", "ANTHROPIC_API_KEY"),
    ("gemini/gemini-2.5-flash", "chat", "GEMINI_API_KEY"),
])
def test_live_canonical_tool_result(model, model_type, key):
    _require_env(key)
    lm = dspy.LM(model, model_type=model_type, engine="lm15", cache=False, num_retries=0)
    request = Request(model=model, messages=(
        Message.user("What is the weather in Paris?"),
        Message.assistant(ToolCallPart(id="call_1", name="weather", input={"city": "Paris"})),
        Message.tool("call_1", "22 C"), Message.user("Repeat the temperature."),
    ), config=Config(max_tokens=256))
    try:
        assert "22" in _text(lm(request))
    finally:
        lm.close()


# ---------------------------------------------------------------------------
# Responses dialect probes.
#
# The Responses request dialect is enforced only by OpenAI's server-side
# validator: the SDK's request types accept shapes the server rejects, and
# mocked tests encode the mapper's own beliefs. One minimal real-API request
# per mapper-supported shape catches the #9943/#9652 class before release.
# Each probe asserts acceptance plus the smallest semantic check; round-trip
# probes cover what acceptance alone can't (e.g. referenceable tool-call ids).
# ---------------------------------------------------------------------------


@pytest.fixture
def responses_lm():
    _require_env("OPENAI_API_KEY")
    return dspy.LM(
        os.getenv("LM_FOR_TEST_RESPONSES", "openai/gpt-5-nano"),
        model_type="responses",
        cache=False,
        temperature=1.0,
        max_tokens=16000,
    )


WEATHER_TOOL_CHAT = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# 1x1 transparent PNG.
TINY_PNG_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _first_tool_call(outputs):
    out = outputs[0]
    assert isinstance(out, dict) and out.get("tool_calls"), f"expected a tool call, got: {out!r}"
    call = out["tool_calls"][0]
    name = call.get("name") or call.get("function", {}).get("name")
    arguments = call.get("arguments") or call.get("function", {}).get("arguments")
    call_id = call.get("call_id") or call.get("id")
    return name, json.loads(arguments), call_id


@pytest.mark.llm_call
def test_probe_history_roles_and_text_content_forms(responses_lm):
    outputs = responses_lm(
        messages=[
            {"role": "developer", "content": "Answer with one word."},
            {"role": "user", "content": [{"type": "text", "text": "Say apple."}]},
            {"role": "assistant", "content": "apple"},
            {"role": "user", "content": "Now say banana."},
            {"role": "assistant", "content": [{"type": "text", "text": "banana"}]},
            {"role": "user", "content": [{"type": "text", "text": "Now say cherry."}]},
        ]
    )
    assert outputs


@pytest.mark.llm_call
def test_probe_image_content(responses_lm):
    outputs = responses_lm(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "One word: what color dominates this image?"},
                    {"type": "image_url", "image_url": {"url": TINY_PNG_URI}},
                ],
            }
        ]
    )
    assert outputs


@pytest.mark.llm_call
def test_probe_flat_tool_with_strict(responses_lm):
    flat_strict = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    outputs = responses_lm(
        "Talk about the weather in Oslo.",
        tools=[flat_strict],
        tool_choice={"type": "function", "name": "get_weather"},
    )
    name, args, _ = _first_tool_call(outputs)
    assert name == "get_weather"
    assert args == {"city": "Oslo"}


@pytest.mark.llm_call
def test_probe_hosted_web_search_tool(responses_lm):
    # The pinned openai SDK models only the "web_search_preview" hosted-tool
    # shape; the newer "web_search" shape fails in litellm's response parsing.
    outputs = responses_lm("Search the web: what year is it? Answer briefly.", tools=[{"type": "web_search_preview"}])
    assert outputs


@pytest.mark.llm_call
def test_probe_tool_choice_none_suppresses_calls(responses_lm):
    outputs = responses_lm(
        "What is the weather in Lima? Use the tool.",
        tools=[WEATHER_TOOL_CHAT],
        tool_choice="none",
    )
    out = outputs[0]
    assert not (isinstance(out, dict) and out.get("tool_calls"))


@pytest.mark.llm_call
def test_probe_response_format_with_reasoning(responses_lm):
    class Answer(pydantic.BaseModel):
        word: str

    outputs = responses_lm("Reply with the word apple.", response_format=Answer, reasoning_effort="low")
    text = outputs[0]["text"] if isinstance(outputs[0], dict) else outputs[0]
    assert Answer.model_validate_json(text).word


@pytest.mark.llm_call
def test_probe_tool_round_trip_ids_are_referenceable(responses_lm):
    """Acceptance alone can't catch id bugs: a request that sends back the
    wrong id (the fc_* item id instead of call_id) fails only on this second
    turn."""
    outputs = responses_lm(
        "Talk about the weather in Berlin.",
        tools=[WEATHER_TOOL_CHAT],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    name, _, call_id = _first_tool_call(outputs)
    tool_calls = outputs[0]["tool_calls"]

    followup = responses_lm(
        messages=[
            {"role": "user", "content": "What is the weather in Berlin? Use the tool."},
            {
                "role": "assistant",
                "content": None,
                # Replay the Responses-shaped output exactly as DSPy returned it.
                "tool_calls": tool_calls,
            },
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": "It is 22C and sunny."},
        ],
        tools=[WEATHER_TOOL_CHAT],
    )
    final = followup[0]
    text = final["text"] if isinstance(final, dict) else final
    assert text and "22" in text
