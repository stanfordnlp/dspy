from unittest import mock

import pytest
from litellm.utils import Choices, Message, ModelResponse

import dspy
from dspy import ChainOfThought
from dspy.utils import DummyLM


def test_initialization_with_string_signature():
    lm = DummyLM([{"reasoning": "find the number after 1", "answer": "2"}])
    dspy.configure(lm=lm)
    predict = ChainOfThought("question -> answer")
    assert list(predict.predict.signature.output_fields.keys()) == [
        "reasoning",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"


@pytest.mark.asyncio
async def test_async_chain_of_thought():
    lm = DummyLM([{"reasoning": "find the number after 1", "answer": "2"}])
    with dspy.context(lm=lm):
        program = ChainOfThought("question -> answer")
        result = await program.acall(question="What is 1+1?")
        assert result.answer == "2"


def test_chain_of_thought_with_native_reasoning():
    """Test ChainOfThought with native reasoning support where LM returns reasoning natively."""

    lm = dspy.LM(model="anthropic/claude-3-7-sonnet-20250219", cache=False)
    dspy.settings.configure(lm=lm)

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="[[ ## answer ## ]]\nParis\n[[ ## completion ## ]]",
                        reasoning_content="Step-by-step thinking about the capital of France",
                    ),
                )
            ],
            model="anthropic/claude-3-7-sonnet-20250219",
        )

        cot = ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert isinstance(result.reasoning, dspy.Reasoning)
        assert result.reasoning.content == "Step-by-step thinking about the capital of France"

        # Check that the reasoning_effort is automatically set to "low" when the LM supports native reasoning and not
        # provided in the LM kwargs
        args, kwargs = mock_completion.call_args
        assert kwargs["reasoning_effort"] == "low"


def test_chain_of_thought_with_manual_reasoning():
    """Test ChainOfThought with manual reasoning where LM doesn't support native reasoning."""
    lm = dspy.LM(model="openai/gpt-4o-mini")
    dspy.settings.configure(lm=lm)

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    reasoning="Step-by-step thinking about the capital of France",
                    message=Message(
                        content=(
                            "[[ ## reasoning ## ]]\nStep-by-step thinking about the capital of France\n"
                            "[[ ## answer ## ]]\nParis\n[[ ## completion ## ]]"
                        )
                    ),
                )
            ],
            model="openai/gpt-4o-mini",
        )

        cot = ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert result.reasoning.content == "Step-by-step thinking about the capital of France"
