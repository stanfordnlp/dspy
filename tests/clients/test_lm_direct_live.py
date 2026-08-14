"""Live coverage for the experimental direct LM call interface.

These tests exercise provider behavior that cannot be verified with the mocked unit tests in `test_lm.py`: typed
message turns, tool-call transcripts, and reusing an `LMResponse` as an assistant turn across chat and Responses API
providers.

They are intentionally flat rather than parametrized so each test can be run individually from an editor or notebook-like
workflow. Each test skips unless the required provider credential is available.
"""

import os

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
