import json
from unittest import mock

import pydantic
import pytest
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from openai.types.responses import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import Summary

import dspy
from dspy.clients.openai_format import responses_to_lm_response, to_openai_responses_request
from dspy.core.types import LMRequest, LMThinkingPart


def _responses_response(output_blocks, *, usage=None):
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="openai/dspy-test-model",
        object="response",
        output=output_blocks,
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=usage or ResponseAPIUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        user=None,
    )


def _chat_shaped_weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


class ContractSchema(pydantic.BaseModel):
    answer: str


def test_openai_format_responses_request_maps_tools_choices_and_config():
    request = LMRequest.from_call(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[_chat_shaped_weather_tool()],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        parallel_tool_calls=False,
        reasoning_effort="low",
        response_format=ContractSchema,
        max_tokens=123,
    )

    data = to_openai_responses_request(request)

    assert data["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the weather in Paris?"}],
        }
    ]
    assert data["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    assert data["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert data["parallel_tool_calls"] is False
    assert data["reasoning"] == {"effort": "low", "summary": "auto"}
    assert data["max_output_tokens"] == 123
    assert data["text"]["format"] == {
        "type": "json_schema",
        "name": "ContractSchema",
        "schema": ContractSchema.model_json_schema(),
    }


def test_responses_to_lm_response_normalizes_mixed_text_reasoning_and_tool_calls():
    response = _responses_response(
        [
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "I should use weather.", "annotations": []}],
            ),
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"city": "Paris",}',
                "call_id": "call_1",
                "id": "fc_1",
                "status": "completed",
            },
            {
                "type": "reasoning",
                "summary": [Summary(type="summary_text", text="Need live weather.")],
            },
        ]
    )

    lm_response = responses_to_lm_response(response, LMRequest(model="openai/dspy-test-model", messages=[]))
    output = lm_response.outputs[0]

    assert output.text == "I should use weather."
    assert output.reasoning_content == "Need live weather."
    assert output.tool_calls[0].id == "call_1"
    assert output.tool_calls[0].name == "get_weather"
    assert output.tool_calls[0].args == {}
    assert output.tool_calls[0].provider_data["raw_arguments"] == '{"city": "Paris",}'
    assert "arguments_parse_error" in output.tool_calls[0].provider_data
    assert lm_response.usage.total_tokens == 3


def test_responses_request_converts_assistant_tool_calls_and_tool_results():
    request = LMRequest.from_call(
        model="openai/gpt-5-mini",
        messages=[
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "sunny"},
        ],
    )

    data = to_openai_responses_request(request)

    assert data["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "What is the weather?"}]},
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Paris"}),
            "call_id": "call_1",
        },
        {"type": "function_call_output", "output": "sunny", "call_id": "call_1"},
    ]


def test_lm_responses_direct_native_tool_calling_uses_responses_tool_shape():
    response = _responses_response(
        [
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": json.dumps({"city": "Paris"}),
                "call_id": "call_1",
                "id": "fc_1",
                "status": "completed",
            }
        ]
    )

    with mock.patch("litellm.responses", autospec=True, return_value=response) as responses:
        lm = dspy.LM("openai/dspy-test-model", model_type="responses", cache=False)
        outputs = lm("What is the weather in Paris?", tools=[_chat_shaped_weather_tool()])

    assert outputs == [
        {
            "text": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                    "id": "call_1",
                }
            ],
        }
    ]
    assert responses.call_args.kwargs["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_lm_responses_async_native_tool_calling_matches_sync_contract():
    response = _responses_response(
        [
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": json.dumps({"city": "Paris"}),
                "call_id": "call_1",
                "id": "fc_1",
                "status": "completed",
            }
        ]
    )

    with mock.patch("litellm.aresponses", autospec=True, return_value=response) as responses:
        lm = dspy.LM("openai/dspy-test-model", model_type="responses", cache=False)
        outputs = await lm.acall("What is the weather in Paris?", tools=[_chat_shaped_weather_tool()])

    assert outputs[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert responses.call_args.kwargs["tools"][0]["name"] == "get_weather"
    assert "function" not in responses.call_args.kwargs["tools"][0]


def test_lm_responses_cache_history_and_usage_use_normalized_output(tmp_path):
    response = _responses_response(
        [
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "cached answer", "annotations": []}],
            ),
            {
                "type": "reasoning",
                "summary": [Summary(type="summary_text", text="cached reasoning")],
            },
        ],
        usage=ResponseAPIUsage(input_tokens=4, output_tokens=5, total_tokens=9),
    )

    original_cache = dspy.cache
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=tmp_path / ".dspy_cache")
    try:
        with mock.patch("litellm.responses", autospec=True, return_value=response) as responses:
            lm = dspy.LM("openai/dspy-test-model", model_type="responses")
            with dspy.context(track_usage=True):
                first = lm("cache me")
                second = lm("cache me")

        assert first == [{"text": "cached answer", "reasoning_content": "cached reasoning"}]
        assert second == first
        assert responses.call_count == 1
        assert lm.history[-1]["outputs"] == first
        assert lm.history[0]["usage"]["input_tokens"] == 4
        assert lm.history[0]["usage"]["output_tokens"] == 5
        assert lm.history[0]["usage"]["total_tokens"] == 9
        assert lm.history[-1]["usage"] == {}
    finally:
        dspy.cache = original_cache


def test_lm_responses_reasoning_output_uses_thinking_part_in_normalized_response():
    response = _responses_response(
        [
            {
                "type": "reasoning",
                "summary": [Summary(type="summary_text", text="think first")],
            }
        ]
    )

    lm_response = responses_to_lm_response(response, LMRequest(model="openai/dspy-test-model", messages=[]))

    assert lm_response.outputs[0].parts == [LMThinkingPart(text="think first")]
    assert lm_response.to_legacy_outputs() == [{"text": None, "reasoning_content": "think first"}]
