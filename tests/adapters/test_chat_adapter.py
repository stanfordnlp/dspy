from typing import Literal
from unittest import mock

import dspy


def test_chat_adapter_quotes_literals_as_expected():
    class TestSignature(dspy.Signature):
        input_text: Literal["one", "two", "three"] = dspy.InputField()
        output_text: Literal["four", "five", "six"] = dspy.OutputField()

    program = dspy.Predict(TestSignature)

    dspy.configure(lm=dspy.LM(model="openai/gpt4o"), adapter=dspy.ChatAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        program(input_text="two")

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert 'Literal["one", "two", "three"]' in str(call_kwargs)
    assert 'Literal["four", "five", "six"]' in str(call_kwargs)
