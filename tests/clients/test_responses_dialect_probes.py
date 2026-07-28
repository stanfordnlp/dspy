"""Live dialect probes for the OpenAI Responses request mapper.

The Responses request dialect is enforced only by OpenAI's server-side
validator: the SDK's request types are demonstrably looser (they accept
assistant `input_text` content, which the server rejects), and mocked unit
tests encode the same beliefs as the mapper they test. These probes fire one
minimal request per mapper-supported request shape at the real API so that
"the server accepts what we emit" is checked systematically instead of
per-incident (#9943 tools, #9652 assistant content).

Each probe asserts acceptance (no invalid-request error) plus the smallest
semantic check that distinguishes success from a silently ignored parameter.
Round-trip probes cover the shapes acceptance alone can't validate, such as
tool-call ids that must be referenceable in the next turn.

Requires OPENAI_API_KEY; skips otherwise. Uses gpt-5-nano.
"""

import base64
import json
import os

import pydantic
import pytest

import dspy

pytestmark = pytest.mark.llm_call


@pytest.fixture
def lm():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Missing live LM credentials: OPENAI_API_KEY")
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
TINY_PNG_URI = "data:image/png;base64," + base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
).decode()


def _first_tool_call(outputs):
    out = outputs[0]
    assert isinstance(out, dict) and out.get("tool_calls"), f"expected a tool call, got: {out!r}"
    call = out["tool_calls"][0]
    name = call.get("name") or call.get("function", {}).get("name")
    arguments = call.get("arguments") or call.get("function", {}).get("arguments")
    call_id = call.get("call_id") or call.get("id")
    return name, json.loads(arguments), call_id


def test_probe_history_all_roles_string_content(lm):
    outputs = lm(
        messages=[
            {"role": "system", "content": "Answer with one word."},
            {"role": "user", "content": "Say apple."},
            {"role": "assistant", "content": "apple"},
            {"role": "user", "content": "Now say banana."},
        ]
    )
    assert outputs


def test_probe_history_list_form_text_content(lm):
    outputs = lm(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Say apple."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "apple"}]},
            {"role": "user", "content": [{"type": "text", "text": "Now say banana."}]},
        ]
    )
    assert outputs


def test_probe_developer_role(lm):
    outputs = lm(
        messages=[
            {"role": "developer", "content": "Answer with one word."},
            {"role": "user", "content": "Say apple."},
        ]
    )
    assert outputs


def test_probe_image_content(lm):
    outputs = lm(
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


def test_probe_nested_chat_tools_return_a_call(lm):
    outputs = lm("What is the weather in Paris? Use the tool.", tools=[WEATHER_TOOL_CHAT])
    name, args, call_id = _first_tool_call(outputs)
    assert name == "get_weather"
    assert args.get("city") == "Paris"
    assert call_id


def test_probe_flat_tool_with_strict(lm):
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
    outputs = lm("What is the weather in Oslo? Use the tool.", tools=[flat_strict])
    name, args, _ = _first_tool_call(outputs)
    assert name == "get_weather"
    assert args == {"city": "Oslo"}


def test_probe_hosted_web_search_tool(lm):
    # The pinned openai SDK models only the "web_search_preview" hosted-tool
    # shape; the newer "web_search" shape fails in litellm's response parsing.
    outputs = lm("Search the web: what year is it? Answer briefly.", tools=[{"type": "web_search_preview"}])
    assert outputs


def test_probe_forced_tool_choice_chat_shape(lm):
    outputs = lm(
        "Talk about the weather in Lima.",
        tools=[WEATHER_TOOL_CHAT],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    name, _, _ = _first_tool_call(outputs)
    assert name == "get_weather"


def test_probe_tool_choice_none_suppresses_calls(lm):
    outputs = lm(
        "What is the weather in Lima? Use the tool.",
        tools=[WEATHER_TOOL_CHAT],
        tool_choice="none",
    )
    out = outputs[0]
    assert not (isinstance(out, dict) and out.get("tool_calls"))


def test_probe_response_format_pydantic(lm):
    class Answer(pydantic.BaseModel):
        word: str

    outputs = lm("Reply with the word apple.", response_format=Answer)
    text = outputs[0]["text"] if isinstance(outputs[0], dict) else outputs[0]
    assert Answer.model_validate_json(text).word


def test_probe_reasoning_effort(lm):
    outputs = lm("Say apple.", reasoning_effort="low")
    assert outputs


def test_probe_tool_round_trip_ids_are_referenceable(lm):
    """Acceptance alone can't catch id bugs: a request that sends back the
    wrong id (the fc_* item id instead of call_id) fails only on this second
    turn."""
    outputs = lm("What is the weather in Berlin? Use the tool.", tools=[WEATHER_TOOL_CHAT])
    name, args, call_id = _first_tool_call(outputs)

    followup = lm(
        messages=[
            {"role": "user", "content": "What is the weather in Berlin? Use the tool."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": "It is 22C and sunny."},
        ],
        tools=[WEATHER_TOOL_CHAT],
    )
    final = followup[0]
    text = final["text"] if isinstance(final, dict) else final
    assert text and "22" in text
