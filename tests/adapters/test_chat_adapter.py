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
            "Literal['one', 'two', 'three\"']",
            "Literal['four', 'five', 'six\"']",
        ),
        # Scenario 2: Single quotes inside strings
        (
            Literal["she's here", "okay", "test"],
            Literal["done", "maybe'soon", "later"],
            "she's here",
            "Literal[\"she's here\", 'okay', 'test']",
            "Literal['done', \"maybe'soon\", 'later']",
        ),
        # Scenario 3: Strings containing both single and double quotes
        (
            Literal["both\"and'", "another"],
            Literal["yet\"another'", "plain"],
            "another",
            "Literal['both\"and\\'', 'another']",
            "Literal['yet\"another\\'', 'plain']",
        ),
        # Scenario 4: No quotes at all (check the default)
        (
            Literal["foo", "bar"],
            Literal["baz", "qux"],
            "foo",
            "Literal['foo', 'bar']",
            "Literal['baz', 'qux']",
        ),
        # Scenario 5: Mixed types
        (
            Literal[1, "bar"],
            Literal[True, 3, "foo"],
            "bar",
            "Literal[1, 'bar']",
            "Literal[True, 3, 'foo']",
        ),
    ],
)
def test_chat_adapter_quotes_literals_as_expected(
    input_literal, output_literal, input_value, expected_input_str, expected_output_str
):
    """
    This test verifies that when we declare Literal fields with various mixes
    of single/double quotes, the generated content string includes those
    Literals exactly as we want them to appear (like IPython does).
    """

    class TestSignature(dspy.Signature):
        input_text: input_literal = dspy.InputField()
        output_text: output_literal = dspy.OutputField()

    program = dspy.Predict(TestSignature)

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.ChatAdapter())

    with mock.patch("litellm.completion") as mock_completion:
        program(input_text=input_value)

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    content = call_kwargs["messages"][0]["content"]

    assert expected_input_str in content
    assert expected_output_str in content
