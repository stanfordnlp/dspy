from typing import Literal
from unittest import mock

import pytest

import dspy


def test_small_lm_adapter_formats_simple_prompt():
    """Test that the first stage prompt is simplified and more natural"""
    
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")
    
    program = dspy.Predict(TestSignature)
    dspy.configure(lm=dspy.LM(model="openai/gpt4"), adapter=dspy.SmallLMAdapter())

    with mock.patch("litellm.completion") as mock_completion:
        program(question="What is 5 + 7?")

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    content = call_kwargs["messages"][0]["content"]

    # Check that the first stage prompt is more natural
    assert "Please solve this math problem" in content
    assert "question:" in content.lower()
    assert "What is 5 + 7?" in content
    assert "Provide a step by step solution and the final numerical answer." in content


def test_small_lm_adapter_extracts_with_json():
    """Test that the second stage uses JSON adapter to extract structured data"""
    
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField(desc="The math question to solve")
        solution: str = dspy.OutputField(desc="Step by step solution")
        answer: float = dspy.OutputField(desc="The final numerical answer")

    # Mock first LM response
    first_response = """
    Let me solve this step by step:
    1. First, we identify the numbers: 5 and 7
    2. Then, we add them together: 5 + 7
    3. The result is 12

    Therefore, 5 + 7 = 12
    """

    adapter = dspy.SmallLMAdapter()
    dspy.configure(adapter=adapter, lm=dspy.LM(model="openai/gpt4"))
    
    # Mock the second stage LM call
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value.choices[0].message.content = """
        {
            "solution": "1. First, we identify the numbers: 5 and 7\\n2. Then, we add them together: 5 + 7\\n3. The result is 12",
            "answer": 12.0
        }
        """
        result = adapter.parse(TestSignature, first_response)
    
    assert result["solution"].startswith("1. First")
    assert result["answer"] == 12.0


def test_small_lm_adapter_handles_complex_types():
    """Test that the adapter can handle more complex output types through JSON extraction"""
    
    class ComplexSignature(dspy.Signature):
        input_text: str = dspy.InputField()
        tags: list[str] = dspy.OutputField(desc="List of relevant tags")
        confidence: float = dspy.OutputField(desc="Confidence score")
    
    # Mock first LM response
    first_response = """
    This text appears to be about machine learning and neural networks.
    I would tag it with: AI, deep learning, and neural networks.
    I'm quite confident about this classification, around 85-90%.
    """
    
    adapter = dspy.SmallLMAdapter()
    dspy.configure(adapter=adapter, lm=dspy.LM(model="openai/gpt4"))
    # Mock the second stage LM call
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value.choices[0].message.content = """
        {
            "tags": ["AI", "deep learning", "neural networks"],
            "confidence": 0.87
        }
        """
        result = adapter.parse(ComplexSignature, first_response)
    
    assert len(result["tags"]) == 3
    assert "AI" in result["tags"]
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1


def test_small_lm_adapter_handles_errors():
    """Test that the adapter properly handles errors in both stages"""
    
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
    
    # Test invalid first stage response
    first_response = "Sorry, I don't know how to help with that."
    
    adapter = dspy.SmallLMAdapter()
    
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value.choices[0].message.content = "{invalid json}"
        with pytest.raises(ValueError, match="Failed to parse response"):
            adapter.parse(TestSignature, first_response) 