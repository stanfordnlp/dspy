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
