from typing import Literal
from unittest import mock

import pytest

import dspy


@pytest.mark.parametrize(
    "input_literal, output_literal, input_value, expected_input_str, expected_output_str",
    [
        # Scenario 1: double quotes escaped within strings
        (
            Literal["one", "two", 'three"'],
            Literal["four", "five", 'six"'],
            "two",
            "<input_text>two</input_text>",
            "must be one of: four; five; six\"",
        ),
        # Scenario 2: Single quotes inside strings
        (
            Literal["she's here", "okay", "test"],
            Literal["done", "maybe'soon", "later"],
            "she's here",
            "<input_text>she's here</input_text>",
            "must be one of: done; maybe'soon; later",
        ),
        # Scenario 3: Strings containing both single and double quotes
        (
            Literal["both\"and'", "another"],
            Literal["yet\"another'", "plain"],
            "another",
            "<input_text>another</input_text>",
            "must be one of: yet\"another'; plain",
        ),
        # Scenario 4: Basic XML parsing test
        (
            Literal["foo", "bar"],
            Literal["baz", "qux"],
            "foo",
            "<input_text>foo</input_text>",
            "must be one of: baz; qux",
        ),
        # Scenario 5: Mixed types
        (
            Literal[1, "bar"],
            Literal[True, 3, "foo"],
            "bar",
            "<input_text>bar</input_text>",
            "must be one of: True; 3; foo",
        ),
    ],
)
def test_xml_adapter_formats_literals_as_expected(
    input_literal, output_literal, input_value, expected_input_str, expected_output_str
):
    """
    This test verifies that when we declare Literal fields, the XML adapter properly
    formats them with XML tags and parses them correctly.
    """

    class TestSignature(dspy.Signature):
        input_text: input_literal = dspy.InputField()
        output_text: output_literal = dspy.OutputField()

    program = dspy.Predict(TestSignature)

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.XMLAdapter())

    with mock.patch("litellm.completion") as mock_completion:
        program(input_text=input_value)

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    content = call_kwargs["messages"][0]["content"]

    assert expected_input_str in content
    assert expected_output_str in content


def test_xml_adapter_basic_parsing():
    """Test that the XML adapter can correctly parse basic XML responses"""
    
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        confidence: float = dspy.OutputField()
    
    xml_response = """
    <answer>This is a test answer</answer>
    <confidence>0.95</confidence>
    <completed></completed>
    """
    
    adapter = dspy.XMLAdapter()
    result = adapter.parse(SimpleSignature, xml_response)
    
    assert result["answer"] == "This is a test answer"
    assert result["confidence"] == 0.95


def test_xml_adapter_handles_nested_xml():
    """Test that the XML adapter properly handles XML content within field values"""
    
    class NestedSignature(dspy.Signature):
        input: str = dspy.InputField()
        output: str = dspy.OutputField()
    
    xml_response = """
    <output>Here is some text with <em>embedded</em> XML that should be preserved</output>
    <completed></completed>
    """
    
    adapter = dspy.XMLAdapter()
    result = adapter.parse(NestedSignature, xml_response)
    
    assert result["output"] == "Here is some text with <em>embedded</em> XML that should be preserved"


def test_xml_adapter_error_handling():
    """Test that the XML adapter properly handles malformed XML"""
    
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
    
    xml_response = """
    <answer>Incomplete tag
    <completed></completed>
    """
    
    adapter = dspy.XMLAdapter()
    with pytest.raises(ValueError, match="Malformed XML"):
        adapter.parse(SimpleSignature, xml_response) 