import pytest

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY
from dspy.utils.dummies import DummyLM


@pytest.fixture(autouse=True)
def clear_history():
    GLOBAL_HISTORY.clear()
    yield


def test_inspect_history_basic(capsys):
    # Configure a DummyLM with some predefined responses
    lm = DummyLM([{"response": "Hello"}, {"response": "How are you?"}])
    dspy.configure(lm=lm)

    # Make some calls to generate history
    predictor = dspy.Predict("query: str -> response: str")
    predictor(query="Hi")
    predictor(query="What's up?")

    # Test inspecting all history
    history = GLOBAL_HISTORY
    print(capsys)
    assert len(history) > 0
    assert isinstance(history, list)
    assert all(isinstance(entry, dict) for entry in history)
    assert all("messages" in entry for entry in history)


def test_inspect_history_with_n(capsys):
    """Test that inspect_history works with n
    Random failures in this test most likely mean you are printing messages somewhere
    """
    lm = DummyLM([{"response": "One"}, {"response": "Two"}, {"response": "Three"}])
    dspy.configure(lm=lm)

    # Generate some history
    predictor = dspy.Predict("query: str -> response: str")
    predictor(query="First")
    predictor(query="Second")
    predictor(query="Third")

    dspy.inspect_history(n=2)
    # Test getting last 2 entries
    out, err = capsys.readouterr()
    assert "First" not in out
    assert "Second" in out
    assert "Third" in out


def test_inspect_empty_history(capsys):
    # Configure fresh DummyLM
    lm = DummyLM([])
    dspy.configure(lm=lm)

    # Test inspecting empty history
    dspy.inspect_history()
    history = GLOBAL_HISTORY
    assert len(history) == 0
    assert isinstance(history, list)


def test_inspect_history_n_larger_than_history(capsys):
    lm = DummyLM([{"response": "First"}, {"response": "Second"}])
    dspy.configure(lm=lm)

    predictor = dspy.Predict("query: str -> response: str")
    predictor(query="Query 1")
    predictor(query="Query 2")

    # Request more entries than exist
    dspy.inspect_history(n=5)
    history = GLOBAL_HISTORY
    assert len(history) == 2  # Should return all available entries


def test_pretty_print_history_handles_tool_calls_only_output(capsys):
    """LM outputs that carry only tool calls have `text` set to None
    (the normalized shape produced by both `_process_completion` and
    `_process_response`). `pretty_print_history` must skip the Response
    section in that case and still print the tool calls."""
    from dspy.utils.inspect_history import pretty_print_history

    entry = {
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "outputs": [
            {
                "text": None,
                "tool_calls": [
                    dspy.ToolCalls.ToolCall(name="get_weather", args={"city": "Paris"}, id="call_1")
                ],
            }
        ],
        "timestamp": "now",
    }

    pretty_print_history([entry], n=1)
    out, _ = capsys.readouterr()
    assert "Tool calls:" in out
    assert "get_weather" in out
    assert "Response:" not in out  # text is None => no Response section


def test_pretty_print_history_displays_tool_role_messages(capsys):
    """Messages with role='tool' (tool results in multi-turn loops) should
    display the tool name, call id, and content."""
    from dspy.utils.inspect_history import pretty_print_history

    entry = {
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}, "id": "call_1"},
            ]},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call_1", "content": "22°C, sunny"},
        ],
        "outputs": [{"text": "It's 22°C and sunny in Paris."}],
        "timestamp": "now",
    }

    pretty_print_history([entry], n=1)
    out, _ = capsys.readouterr()
    assert "Tool message:" in out
    assert "get_weather" in out
    assert "call_1" in out
    assert "22°C, sunny" in out
    assert "Assistant message:" in out
