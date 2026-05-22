import pytest

import dspy


def test_anthropic_class_maps_request_and_response():
    calls = []

    def requester(payload, stream):
        calls.append((payload, stream))
        return {
            "id": "msg_1",
            "model": "claude-3-5-haiku-latest",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }

    lm = dspy.AnthropicLM("anthropic/claude-3-5-haiku-latest", requester=requester, cache=False)

    response = lm("say hello", temperature=0.2)

    payload, stream = calls[0]
    assert stream is False
    assert payload["model"] == "claude-3-5-haiku-latest"
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "say hello"}]}]
    assert payload["temperature"] == 0.2
    assert response.text == "hello"
    assert response.usage.input_tokens == 2
    assert response.usage.output_tokens == 3


def test_anthropic_class_maps_tools_and_tool_results():
    calls = []

    def requester(payload, stream):
        calls.append(payload)
        return {
            "id": "msg_1",
            "model": "claude-3-5-haiku-latest",
            "content": [{"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "DSPy"}}],
            "stop_reason": "tool_use",
            "usage": {},
        }

    lm = dspy.AnthropicLM("anthropic/claude-3-5-haiku-latest", requester=requester, cache=False)
    request = dspy.LMRequest.from_call(
        model="anthropic/claude-3-5-haiku-latest",
        items=(
            dspy.User("search"),
            dspy.Assistant(dspy.LMToolCall(id="tool_1", name="search", args={"q": "DSPy"})),
            dspy.ToolResult("result", call_id="tool_1", name="search"),
        ),
        tools=[dspy.lm.LMToolSpec(name="search", description="Search", parameters={"type": "object"})],
        tool_choice="required",
    )

    response = lm(request)

    assert calls[0]["tools"] == [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
    assert calls[0]["tool_choice"] == {"type": "any"}
    assert calls[0]["messages"][1]["content"][0]["type"] == "tool_use"
    assert calls[0]["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].args == {"q": "DSPy"}


def test_genai_class_maps_request_and_response():
    calls = []

    def requester(payload, stream):
        calls.append((payload, stream))
        return {
            "responseId": "resp_1",
            "candidates": [
                {
                    "content": {"parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
        }

    lm = dspy.GenAILM("gemini/gemini-2.0-flash", requester=requester, cache=False)

    response = lm("say hello", max_tokens=10)

    payload, stream = calls[0]
    assert stream is False
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "say hello"}]}]
    assert payload["generationConfig"]["maxOutputTokens"] == 10
    assert response.text == "hello"
    assert response.usage.total_tokens == 5


def test_anthropic_stream_emits_start_before_delta_when_provider_omits_message_start():
    def requester(payload, stream):
        assert stream is True
        return [
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]

    lm = dspy.AnthropicLM("anthropic/claude-3-5-haiku-latest", requester=requester, cache=False)

    events = list(lm.stream("say hello"))

    assert events[0].type == "start"
    assert events[1].type == "delta"


@pytest.mark.asyncio
async def test_anthropic_async_call_and_stream_use_anyio_thread_bridge():
    def requester(payload, stream):
        if stream:
            return [
                {"type": "message_start", "message": {"model": "claude-3-5-haiku-latest"}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 1, "output_tokens": 1}},
                {"type": "message_stop"},
            ]
        return {
            "id": "msg_1",
            "model": "claude-3-5-haiku-latest",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    lm = dspy.AnthropicLM("anthropic/claude-3-5-haiku-latest", requester=requester, cache=False)

    response = await lm.acall("say hello")
    stream = lm.astream("say hello")
    events = [event async for event in stream]

    assert response.text == "hello"
    assert events[-1].type == "end"
    assert stream.result().text == "hello"


@pytest.mark.asyncio
async def test_genai_async_call_and_stream_use_anyio_thread_bridge():
    def requester(payload, stream):
        if stream:
            return [
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "hello"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                }
            ]
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        }

    lm = dspy.GenAILM("gemini/gemini-2.0-flash", requester=requester, cache=False)

    response = await lm.acall("say hello")
    stream = lm.astream("say hello")
    events = [event async for event in stream]

    assert response.text == "hello"
    assert events[-1].type == "end"
    assert stream.result().text == "hello"


def test_router_returns_direct_anthropic_and_genai_backends():
    with dspy.context(experimental=True):
        anthropic = dspy.LM("anthropic/claude-3-5-haiku-latest", cache=False)
        gemini = dspy.LM("gemini/gemini-2.0-flash", cache=False)

    assert isinstance(anthropic, dspy.AnthropicLM)
    assert isinstance(gemini, dspy.GenAILM)
