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


def test_pretty_print_history_renders_assistant_message_tool_calls():
    """An assistant message in the messages list with `tool_calls` (and typically
    `content=None`) must render the calls under a `Tool calls:` header. Without
    coverage here, a regression in the messages-loop tool_calls rendering branch
    would go undetected."""
    history = [
        {
            "timestamp": "now",
            "messages": [
                {"role": "user", "content": "search for it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"q":"dspy"}'},
                        }
                    ],
                },
            ],
            "outputs": [{"text": "done", "tool_calls": None}],
        }
    ]
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "Tool calls:" in output
    assert "web_search" in output
    assert '{"q":"dspy"}' in output


def test_pretty_print_history_renders_tool_role_message_with_tool_call_id():
    """A `tool` role message must render with its `tool_call_id` in the role label
    so the user can match each tool result back to the call that produced it."""
    history = [
        {
            "timestamp": "now",
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_xyz",
                    "content": "search result body",
                },
            ],
            "outputs": [{"text": "final answer", "tool_calls": None}],
        }
    ]
    buf = io.StringIO()
    pretty_print_history(history, n=1, file=buf)
    output = buf.getvalue()
    assert "tool_call_id=call_xyz" in output
    assert "search result body" in output
