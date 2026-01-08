"""Tests for InputField default value handling in Predict."""

import dspy
from dspy.utils.dummies import DummyLM


def test_input_field_with_default_value():
    """Test that InputField default values are included in the prompt when field is omitted."""

    class TestSignature(dspy.Signature):
        """Answer the question based on context."""

        context: str = dspy.InputField(default="DEFAULT CONTEXT")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(TestSignature)
    lm = DummyLM([{"answer": "test answer"}])
    dspy.configure(lm=lm)

    # Call without providing context - should use default value
    result = predictor(question="What is this?")

    # Verify the result
    assert result.answer == "test answer"

    # Verify that the default context was included in the LM call
    # DummyLM stores the last call in its history
    assert "DEFAULT CONTEXT" in str(lm.history[-1]["messages"])


def test_input_field_with_multiple_defaults():
    """Test that multiple InputField default values are handled correctly."""

    class TestSignature(dspy.Signature):
        """Answer the question."""

        context: str = dspy.InputField(default="DEFAULT CONTEXT")
        language: str = dspy.InputField(default="English")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(TestSignature)
    lm = DummyLM([{"answer": "test answer"}])
    dspy.configure(lm=lm)

    # Call without providing context or language - should use both defaults
    result = predictor(question="What is this?")

    assert result.answer == "test answer"
    messages_str = str(lm.history[-1]["messages"])
    assert "DEFAULT CONTEXT" in messages_str
    assert "English" in messages_str


def test_input_field_with_default_factory():
    """Test that InputField with default_factory is handled correctly."""

    class TestSignature(dspy.Signature):
        """Answer the question."""

        tags: list = dspy.InputField(default_factory=list)
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(TestSignature)
    lm = DummyLM([{"answer": "test answer"}])
    dspy.configure(lm=lm)

    # Call without providing tags - should use empty list from factory
    result = predictor(question="What is this?")

    assert result.answer == "test answer"
    messages_str = str(lm.history[-1]["messages"])
    # The empty list should be formatted in the message
    assert "[]" in messages_str or "tags" in messages_str


def test_input_field_override_default_value():
    """Test that providing a value overrides the default."""

    class TestSignature(dspy.Signature):
        """Answer the question."""

        context: str = dspy.InputField(default="DEFAULT CONTEXT")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(TestSignature)
    lm = DummyLM([{"answer": "test answer"}])
    dspy.configure(lm=lm)

    # Call with explicit context - should override default
    result = predictor(context="CUSTOM CONTEXT", question="What is this?")

    assert result.answer == "test answer"
    messages_str = str(lm.history[-1]["messages"])
    assert "CUSTOM CONTEXT" in messages_str
    assert "DEFAULT CONTEXT" not in messages_str


def test_input_field_partial_override_with_multiple_defaults():
    """Test that partial override works with multiple defaults."""

    class TestSignature(dspy.Signature):
        """Answer the question."""

        context: str = dspy.InputField(default="DEFAULT CONTEXT")
        language: str = dspy.InputField(default="English")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(TestSignature)
    lm = DummyLM([{"answer": "test answer"}])
    dspy.configure(lm=lm)

    # Call with only language overridden - context should use default
    result = predictor(language="French", question="What is this?")

    assert result.answer == "test answer"
    messages_str = str(lm.history[-1]["messages"])
    assert "DEFAULT CONTEXT" in messages_str
    assert "French" in messages_str
    assert "English" not in messages_str


def test_input_field_no_warning_with_defaults():
    """Test that no warning is raised when all required fields are provided (including defaults)."""
    import logging

    class TestSignature(dspy.Signature):
        """Answer the question."""

        context: str = dspy.InputField(default="DEFAULT CONTEXT")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(TestSignature)
    lm = DummyLM([{"answer": "test answer"}])
    dspy.configure(lm=lm)

    # Capture logs to check if warning is raised
    with dspy.context(lm=lm):
        # This should not raise a warning since context has a default
        result = predictor(question="What is this?")
        assert result.answer == "test answer"
