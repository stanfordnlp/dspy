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
