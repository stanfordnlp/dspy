import dspy
from dspy.predict.reactv2 import ReActV2, ToolExecutionResult, _build_submit_tool
from dspy.utils.dummies import DummyLM


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_submit_tool_returns_dict():
    signature = dspy.Signature("question -> answer")
    submit = _build_submit_tool(signature)

    assert submit(answer="42") == {"answer": "42"}


def test_basic_forward_with_submit_records_history_messages():
    lm = DummyLM(
        [
            {"next_thought": "I should add.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
            {"next_thought": "I have the answer.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
        ]
    )
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[add])

    result = react(question="What is 1+2?")

    assert result.answer == "3"
    assert len(result.history.messages) == 2
    assert result.history.messages[0]["question"] == "What is 1+2?"
    assert result.history.messages[0]["next_thought"] == "I should add."
    assert result.history.messages[0]["tool_call_results"].tool_call_results[0].value == 3
    assert result.history.messages[-1]["answer"] == "3"
    second_call_user_messages = [message["content"] for message in lm.history[1]["messages"] if message["role"] == "user"]
    assert any("tool_call_results" in content for content in second_call_user_messages)
    assert all("None" not in content for content in second_call_user_messages)
    assert "What is 1+2?" in second_call_user_messages[0]
    assert "What is 1+2?" not in second_call_user_messages[-1]


def test_forward_with_native_tool_calling_renders_tool_results_as_tool_messages():
    class NativeToolLoopLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="openai/gpt-5-nano", cache=False)
            self.calls = []

        @property
        def supported_params(self):
            return frozenset()

        @property
        def supports_function_calling(self):
            return True

        @property
        def supports_reasoning(self):
            return False

        @property
        def supports_response_schema(self):
            return False

        def __call__(self, messages, **kwargs):
            self.calls.append({"messages": messages, "kwargs": kwargs})
            if len(self.calls) == 1:
                return [
                    {
                        "text": "[[ ## next_thought ## ]]\nI should add.\n\n[[ ## completed ## ]]",
                        "tool_calls": [
                            {
                                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                                "id": "call_add",
                                "type": "function",
                            }
                        ],
                    }
                ]
            return [
                {
                    "text": "[[ ## next_thought ## ]]\nI can answer.\n\n[[ ## completed ## ]]",
                    "tool_calls": [
                        {
                            "function": {"name": "submit", "arguments": '{"answer": "3"}'},
                            "id": "call_submit",
                            "type": "function",
                        }
                    ],
                }
            ]

    lm = NativeToolLoopLM()
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True)):
        result = ReActV2("question -> answer", tools=[add])(question="What is 1+2?")

    assert result.answer == "3"
    assert result.history.messages[0]["tool_call_results"].tool_call_results[0].call_id == "call_add"
    assert any(message["role"] == "tool" and message["tool_call_id"] == "call_add" for message in lm.calls[1]["messages"])
    assert lm.calls[0]["kwargs"]["tools"][0]["function"]["name"] == "add"


def test_react_signature_keeps_optional_tool_call_results_input():
    react = ReActV2("question -> answer", tools=[add])

    assert react.react.signature.input_fields["question"].default is None
    field = react.react.signature.input_fields["tool_call_results"]
    assert field.default is None
    assert field.annotation == dspy.ToolCallResults | None


def test_forward_with_existing_history_does_not_append_empty_input_event():
    lm = DummyLM(
        [
            {"next_thought": "I have the answer.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
        ]
    )
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[add])
    history = dspy.History(messages=[{"question": "What is 1+2?"}])

    result = react(history=history)

    assert result.answer == "3"
    assert {} not in result.history.messages
    assert result.history.messages[0] == {"question": "What is 1+2?"}


def test_unknown_tool_returns_error_observation():
    lm = DummyLM(
        [
            {"next_thought": "Call fake.", "tool_calls": [{"name": "nonexistent", "args": {}}]},
            {"next_thought": "Now submit.", "tool_calls": [{"name": "submit", "args": {"answer": "ok"}}]},
        ]
    )
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[add])

    result = react(question="test")

    assert result.answer == "ok"
    tool_results = [
        result
        for message in result.history.messages
        for result in getattr(message.get("tool_call_results"), "tool_call_results", [])
    ]
    assert any(result.is_error and "Unknown tool" in str(result.value) for result in tool_results)


def test_append_tool_turn_records_observation_ids():
    history = dspy.History(messages=[])
    tool_calls = dspy.ToolCalls.from_dict_list([{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_add"}])

    ReActV2._append_tool_turn(
        history,
        next_thought="add",
        tool_calls=tool_calls,
        tool_results=[ToolExecutionResult(value=3)],
    )

    tool_result = history.messages[0]["tool_call_results"].tool_call_results[0]
    assert tool_result.call_id == "call_add"
    assert tool_result.name == "add"


def test_forced_submit_runs_when_loop_returns_no_tool_calls():
    lm = DummyLM(
        [
            {"next_thought": "No call.", "tool_calls": []},
            {"next_thought": "Force submit.", "tool_calls": [{"name": "submit", "args": {"answer": "done"}}]},
        ]
    )
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[add])

    result = react(question="test")

    assert result.answer == "done"
    assert result.termination_reason == "forced_submit"
