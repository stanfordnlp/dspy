"""Live coverage for the experimental direct LM call interface.

These tests exercise provider behavior that cannot be verified with the mocked unit tests in `test_lm.py`: typed
message turns, tool-call transcripts, and reusing an `LMResponse` as an assistant turn across chat and Responses API
providers.

They are intentionally flat rather than parametrized so each test can be run individually from an editor or notebook-like
workflow. Each test skips unless the required provider credential is available.
"""

import json
import os

import pydantic
import pytest

import dspy


def _require_env(*keys: str) -> None:
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing live LM credentials: {', '.join(missing)}")


def _text(response: dspy.LMResponse) -> str:
    assert isinstance(response, dspy.LMResponse)
    assert response.text is not None
    return response.text.strip()


@pytest.mark.llm_call
def test_live_openai_chat_direct_system_user_assistant_multiturn():
    _require_env("OPENAI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_OPENAI_CHAT", "openai/gpt-5.5"),
        model_type="chat",
        cache=False,
        max_completion_tokens=64,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the user's requested exact final token. No punctuation."),
            dspy.User("Reply with exactly: alpha"),
            dspy.Assistant("alpha"),
            dspy.User("Now reply with exactly: beta"),
        )

    assert "beta" in _text(response).lower()
    assert response.output.finish_reason is not None


@pytest.mark.llm_call
def test_live_openai_chat_direct_tool_call_transcript():
    _require_env("OPENAI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_OPENAI_CHAT", "openai/gpt-4o-mini"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Use the supplied tool result. Keep the answer short."),
            dspy.User("What is the weather in Paris?"),
            dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
            dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
            dspy.User("Answer with the temperature string from the tool result."),
        )

    text = _text(response).lower()
    assert "22" in text
    assert "c" in text


@pytest.mark.llm_call
def test_live_openai_chat_direct_reuse_lm_response_as_assistant_turn():
    _require_env("OPENAI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_OPENAI_CHAT", "openai/gpt-4o-mini"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        first = lm(dspy.User("Reply with exactly: DSPy"))
        follow_up = lm(
            dspy.User("Reply with exactly: DSPy"),
            first,
            dspy.User("Repeat the previous assistant answer exactly."),
        )

    assert "dspy" in _text(first).lower()
    assert "dspy" in _text(follow_up).lower()


@pytest.mark.llm_call
def test_live_openai_responses_direct_system_user_assistant_multiturn():
    _require_env("OPENAI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_OPENAI_RESPONSES", "openai/gpt-4.1-mini"),
        model_type="responses",
        cache=False,
        temperature=1.0,
        max_tokens=16000,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the user's requested exact final token. No punctuation."),
            dspy.User("Reply with exactly: alpha"),
            dspy.Assistant("alpha"),
            dspy.User("Now reply with exactly: beta"),
        )

    assert "beta" in _text(response).lower()


@pytest.mark.llm_call
def test_live_openai_responses_direct_tool_call_transcript():
    _require_env("OPENAI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_OPENAI_RESPONSES", "openai/gpt-4.1-mini"),
        model_type="responses",
        cache=False,
        temperature=1.0,
        max_tokens=16000,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Use the supplied tool result. Keep the answer short."),
            dspy.User("What is the weather in Paris?"),
            dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
            dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
            dspy.User("Answer with the temperature string from the tool result."),
        )

    text = _text(response).lower()
    assert "22" in text
    assert "c" in text


@pytest.mark.llm_call
def test_live_openai_responses_direct_reuse_lm_response_as_assistant_turn():
    _require_env("OPENAI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_OPENAI_RESPONSES", "openai/gpt-4.1-mini"),
        model_type="responses",
        cache=False,
        temperature=1.0,
        max_tokens=16000,
    )

    with dspy.context(experimental=True):
        first = lm(dspy.User("Reply with exactly: DSPy"))
        follow_up = lm(
            dspy.User("Reply with exactly: DSPy"),
            first,
            dspy.User("Repeat the previous assistant answer exactly."),
        )

    assert "dspy" in _text(first).lower()
    assert "dspy" in _text(follow_up).lower()


@pytest.mark.llm_call
def test_live_anthropic_chat_direct_system_user_assistant_multiturn():
    _require_env("ANTHROPIC_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_ANTHROPIC", "anthropic/claude-3-5-haiku-latest"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the user's requested exact final token. No punctuation."),
            dspy.User("Reply with exactly: alpha"),
            dspy.Assistant("alpha"),
            dspy.User("Now reply with exactly: beta"),
        )

    assert "beta" in _text(response).lower()


@pytest.mark.llm_call
def test_live_anthropic_chat_direct_tool_call_transcript():
    _require_env("ANTHROPIC_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_ANTHROPIC", "anthropic/claude-3-5-haiku-latest"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Use the supplied tool result. Keep the answer short."),
            dspy.User("What is the weather in Paris?"),
            dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
            dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
            dspy.User("Answer with the temperature string from the tool result."),
        )

    text = _text(response).lower()
    assert "22" in text
    assert "c" in text


@pytest.mark.llm_call
def test_live_anthropic_chat_direct_reuse_lm_response_as_assistant_turn():
    _require_env("ANTHROPIC_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_ANTHROPIC", "anthropic/claude-3-5-haiku-latest"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        first = lm(dspy.User("Reply with exactly: DSPy"))
        follow_up = lm(
            dspy.User("Reply with exactly: DSPy"),
            first,
            dspy.User("Repeat the previous assistant answer exactly."),
        )

    assert "dspy" in _text(first).lower()
    assert "dspy" in _text(follow_up).lower()


@pytest.mark.llm_call
def test_live_gemini_chat_direct_system_user_assistant_multiturn():
    _require_env("GEMINI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_GEMINI", "gemini/gemini-2.0-flash"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the user's requested exact final token. No punctuation."),
            dspy.User("Reply with exactly: alpha"),
            dspy.Assistant("alpha"),
            dspy.User("Now reply with exactly: beta"),
        )

    assert "beta" in _text(response).lower()


@pytest.mark.llm_call
def test_live_gemini_chat_direct_tool_call_transcript():
    _require_env("GEMINI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_GEMINI", "gemini/gemini-2.0-flash"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Use the supplied tool result. Keep the answer short."),
            dspy.User("What is the weather in Paris?"),
            dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
            dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
            dspy.User("Answer with the temperature string from the tool result."),
        )

    text = _text(response).lower()
    assert "22" in text
    assert "c" in text


@pytest.mark.llm_call
def test_live_gemini_chat_direct_reuse_lm_response_as_assistant_turn():
    _require_env("GEMINI_API_KEY")

    lm = dspy.LM(
        os.getenv("LM_FOR_TEST_DIRECT_GEMINI", "gemini/gemini-2.0-flash"),
        model_type="chat",
        cache=False,
        max_tokens=64,
    )

    with dspy.context(experimental=True):
        first = lm(dspy.User("Reply with exactly: DSPy"))
        follow_up = lm(
            dspy.User("Reply with exactly: DSPy"),
            first,
            dspy.User("Repeat the previous assistant answer exactly."),
        )

    assert "dspy" in _text(first).lower()
    assert "dspy" in _text(follow_up).lower()


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
