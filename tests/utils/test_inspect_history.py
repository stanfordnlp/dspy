import io

from dspy.adapters.types.tool import ToolCalls
from dspy.utils.inspect_history import _format_tool_call, pretty_print_history


def test_format_tool_call_openai_chat_completions_shape():
    """OpenAI Chat Completions wire format: {'function': {'name', 'arguments'}}."""
    item = {"function": {"name": "search", "arguments": '{"q":"x"}'}}
    assert _format_tool_call(item) == 'search: {"q":"x"}'


def test_format_tool_call_openai_responses_api_shape():
    """OpenAI Responses API shape: name + arguments on the item itself."""
    item = {"name": "search", "arguments": '{"q":"y"}'}
    assert _format_tool_call(item) == 'search: {"q":"y"}'


def test_format_tool_call_dspy_native_toolcall_shape():
    """DSPy-native ToolCall pydantic object uses `args` (dict), not `function.arguments`."""
    tc = ToolCalls.ToolCall(name="lookup", args={"key": "val"})
    result = _format_tool_call(tc)
    assert "lookup" in result and "val" in result


def test_pretty_print_history_renders_dspy_native_tool_call():
    """A history entry whose tool_calls contain DSPy-native ToolCall objects should
    render the name and args, not produce empty `: ` output."""
    history = [
        {
            "timestamp": "now",
            "messages": [{"role": "user", "content": "hi"}],
            "outputs": [
                {
                    "text": "",
                    "tool_calls": [ToolCalls.ToolCall(name="lookup", args={"key": "val"})],
                }
            ],
        }
    ]
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "lookup" in output
    assert "val" in output
