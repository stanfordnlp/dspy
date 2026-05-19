import pytest

import dspy


class RecordingLM:
    supports_reasoning = False
    supports_response_schema = True

    def __init__(self, outputs, supported_params=None):
        self.outputs = outputs
        self.supported_params = set(supported_params or [])
        self.supports_function_calling = True
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.outputs

    async def acall(self, messages, **kwargs):
        return self(messages, **kwargs)


class ToolSignature(dspy.Signature):
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    answer: str = dspy.OutputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"


def get_timezone(city: str) -> str:
    return f"The timezone in {city} is CET"


TOOLS = [dspy.Tool(get_weather), dspy.Tool(get_timezone)]
EXPECTED_TOOL_CALLS = dspy.ToolCalls(
    tool_calls=[
        dspy.ToolCalls.ToolCall(name="get_weather", args={"city": "Paris"}),
        dspy.ToolCalls.ToolCall(name="get_timezone", args={"city": "Paris"}),
    ]
)


def _supported_params(adapter_cls):
    return {"response_format"} if adapter_cls is dspy.JSONAdapter else set()


def _completion(adapter_cls, *, tool_calls=None):
    if adapter_cls is dspy.JSONAdapter:
        if tool_calls is None:
            return '{"answer": "ok"}'
        return f'{{"answer": "Need tools", "tool_calls": {tool_calls}}}'

    if tool_calls is None:
        return "[[ ## answer ## ]]\nok\n\n[[ ## completed ## ]]"
    return f"[[ ## answer ## ]]\nNeed tools\n\n[[ ## tool_calls ## ]]\n{tool_calls}\n\n[[ ## completed ## ]]"


def _run(adapter, lm):
    return adapter(
        lm,
        {},
        ToolSignature,
        [],
        {"question": "What is the weather in Paris?", "tools": TOOLS},
    )


@pytest.mark.parametrize("adapter_cls", [dspy.ChatAdapter, dspy.JSONAdapter])
@pytest.mark.parametrize("allow_parallel_tool_calls", [None, False, True])
def test_native_tool_call_request_uses_native_surface(adapter_cls, allow_parallel_tool_calls):
    adapter = adapter_cls(
        use_native_function_calling=True,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
    )
    lm = RecordingLM([_completion(adapter_cls)], supported_params=_supported_params(adapter_cls))

    result = _run(adapter, lm)
    call = lm.calls[0]
    system_message = call["messages"][0]["content"]

    assert result[0]["answer"] == "ok"
    assert result[0]["tool_calls"] is None
    assert "tools" in call["kwargs"]
    assert "native tool-call interface" in system_message
    assert "[[ ## tools ## ]]" not in system_message
    assert "[[ ## tool_calls ## ]]" not in system_message
    if allow_parallel_tool_calls is None:
        assert "parallel_tool_calls" not in call["kwargs"]
    else:
        assert call["kwargs"]["parallel_tool_calls"] is allow_parallel_tool_calls
    if adapter_cls is dspy.JSONAdapter:
        assert set(call["kwargs"]["response_format"].model_json_schema()["properties"]) == {"answer"}


@pytest.mark.parametrize("adapter_cls", [dspy.ChatAdapter, dspy.JSONAdapter])
def test_non_native_tool_calls_parse_multiple_calls(adapter_cls):
    adapter = adapter_cls(use_native_function_calling=False, allow_parallel_tool_calls=True)
    tool_calls = """
[
  {"name": "get_weather", "args": {"city": "Paris"}},
  {"name": "get_timezone", "args": {"city": "Paris"}}
]
    """
    lm = RecordingLM([_completion(adapter_cls, tool_calls=tool_calls)], supported_params=_supported_params(adapter_cls))

    result = _run(adapter, lm)[0]
    call = lm.calls[0]

    assert result["answer"] == "Need tools"
    assert result["tool_calls"] == EXPECTED_TOOL_CALLS
    assert "tools" not in call["kwargs"]
    assert "parallel_tool_calls" not in call["kwargs"]
    assert "tool_calls" in call["messages"][0]["content"]
    if adapter_cls is dspy.JSONAdapter:
        assert call["kwargs"]["response_format"] == {"type": "json_object"}


def test_native_function_calling_preserves_multiple_chat_tool_calls():
    lm = RecordingLM(
        [
            {
                "text": None,
                "tool_calls": [
                    {
                        "function": {"arguments": '{"city":"Paris"}', "name": "get_weather"},
                        "id": "call_weather",
                        "type": "function",
                    },
                    {
                        "function": {"arguments": '{"city":"Paris"}', "name": "get_timezone"},
                        "id": "call_timezone",
                        "type": "function",
                    },
                ],
            }
        ]
    )

    result = _run(dspy.ChatAdapter(use_native_function_calling=True), lm)[0]

    assert result["answer"] is None
    assert result["tool_calls"] == EXPECTED_TOOL_CALLS


def test_native_function_calling_parses_responses_api_tool_call_shape():
    lm = RecordingLM(
        [
            {
                "tool_calls": [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    }
                ]
            }
        ]
    )

    result = _run(dspy.ChatAdapter(use_native_function_calling=True), lm)[0]

    assert result["answer"] is None
    assert result["tool_calls"] == dspy.ToolCalls(
        tool_calls=[dspy.ToolCalls.ToolCall(name="get_weather", args={"city": "Paris"})]
    )


def test_legacy_custom_type_native_hook_signature_still_works():
    class CustomType(dspy.Type):
        value: str

        @classmethod
        def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
            lm_kwargs["legacy_hook_called"] = True
            return signature.delete(field_name)

        @classmethod
        def parse_lm_response(cls, response):
            if isinstance(response, dict) and "custom_value" in response:
                return cls(value=response["custom_value"])
            return None

    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: CustomType = dspy.OutputField()

    lm = RecordingLM([{"text": None, "custom_value": "ok"}])
    result = dspy.ChatAdapter(native_response_types=[CustomType])(
        lm,
        {},
        MySignature,
        [],
        {"question": "hello"},
    )

    assert result[0]["answer"] == CustomType(value="ok")
    assert lm.calls[0]["kwargs"]["legacy_hook_called"] is True
