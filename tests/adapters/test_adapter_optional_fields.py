from typing import Annotated, Optional

import dspy
from dspy.adapters.base import Adapter


def test_adapter_detects_optional_history_and_tools():
    class OptionalHistory(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History | None = dspy.InputField()
        answer: str = dspy.OutputField()

    class TypingOptionalHistory(dspy.Signature):
        question: str = dspy.InputField()
        history: Optional[dspy.History] = dspy.InputField()
        answer: str = dspy.OutputField()

    class OptionalTools(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] | None = dspy.InputField()
        answer: str = dspy.OutputField()

    class AnnotatedHistory(dspy.Signature):
        question: str = dspy.InputField()
        history: Annotated[dspy.History, "chat"] = dspy.InputField()
        answer: str = dspy.OutputField()

    class OptionalToolCalls(dspy.Signature):
        question: str = dspy.InputField()
        tool_calls: dspy.ToolCalls | None = dspy.OutputField()

    adapter = Adapter()
    assert adapter._get_history_field_name(OptionalHistory) == "history"
    assert adapter._get_history_field_name(TypingOptionalHistory) == "history"
    assert adapter._get_history_field_name(AnnotatedHistory) == "history"
    assert adapter._get_tool_call_input_field_name(OptionalTools) == "tools"
    assert adapter._get_tool_call_output_field_name(OptionalToolCalls) == "tool_calls"


def test_optional_tool_calls_are_omitted_from_native_history_content():
    from dspy.adapters.types.tool import ToolCallResults

    class OptionalToolCalls(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History | None = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls | None = dspy.OutputField()

    adapter = dspy.ChatAdapter(use_native_function_calling=True)
    tool_calls = dspy.ToolCalls.from_dict_list(
        [{"name": "search", "args": {"query": "cats"}, "id": "call_1"}]
    )
    tool_calls = tool_calls.model_copy(
        update={"tool_call_results": ToolCallResults.from_tool_calls_and_values(tool_calls, ["meow"])}
    )
    history = dspy.History(
        messages=[{"question": "find cats", "answer": "searching", "tool_calls": tool_calls}]
    )
    inputs = {"history": history, "question": "next"}
    messages = adapter.format_conversation_history(OptionalToolCalls, "history", inputs)

    assistant = next(message for message in messages if message.get("role") == "assistant")
    content = assistant.get("content") or ""
    assert "[[ ## tool_calls ## ]]" not in content
    assert assistant.get("tool_calls")
