from dataclasses import dataclass
from unittest import mock

import pytest
from litellm.utils import Choices, Message, ModelResponse
from pydantic import BaseModel

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
    """Test ChainOfThought with a model that supports native reasoning, but using manual fields."""

    lm = dspy.LM(model="anthropic/claude-3-7-sonnet-20250219", cache=False)
    dspy.settings.configure(lm=lm)

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content=(
                            "[[ ## reasoning ## ]]\nStep-by-step thinking about the capital of France\n"
                            "[[ ## answer ## ]]\nParis\n[[ ## completion ## ]]"
                        )
                    ),
                )
            ],
            model="anthropic/claude-3-7-sonnet-20250219",
        )

        cot = ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert isinstance(result.reasoning, str)
        assert result.reasoning == "Step-by-step thinking about the capital of France"

        args, kwargs = mock_completion.call_args


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
        assert result.reasoning == "Step-by-step thinking about the capital of France"


def test_chain_of_thought_with_typed_dataclass_signature():
    """Test ChainOfThought with typed signatures using dataclasses."""

    @dataclass
    class QuestionInput:
        question: str

    class AnswerOutput:
        reasoning: str
        answer: float

    lm = DummyLM([{"reasoning": "1+1 equals 2", "answer": 2}])
    dspy.configure(lm=lm)

    # Create ChainOfThought with typed signature
    cot = ChainOfThought(dspy.Signature(input_type=QuestionInput, output_type=AnswerOutput))

    # Test with typed input
    result = cot(QuestionInput(question="What is 1+1?"))

    assert isinstance(result, AnswerOutput)
    assert isinstance(result.answer, float)
    assert result.answer == 2
    assert result.reasoning == "1+1 equals 2"


def test_chain_of_thought_with_typed_pydantic_signature():
    """Test ChainOfThought with typed signatures using Pydantic models."""

    class QueryInput(BaseModel):
        question: str
        context: str = "basic arithmetic"

    class ReasonedOutput(BaseModel):
        reasoning: str
        answer: str

    lm = DummyLM([{"reasoning": "Following the rules of addition", "answer": "3"}])
    dspy.configure(lm=lm)

    # Create ChainOfThought with typed signature
    cot = ChainOfThought(dspy.Signature(input_type=QueryInput, output_type=ReasonedOutput))

    # Test with typed input
    result = cot(QueryInput(question="What is 2+1?", context="basic arithmetic"))

    assert isinstance(result, ReasonedOutput)
    assert result.answer == "3"
    assert result.reasoning == "Following the rules of addition"


@pytest.mark.asyncio
async def test_async_chain_of_thought_with_typed_signature():
    """Test async ChainOfThought with typed signatures."""

    @dataclass
    class ProblemInput:
        problem: str

    class SolutionOutput:
        reasoning: str
        solution: str

    lm = DummyLM([{"reasoning": "Logical deduction", "solution": "Yes"}])

    with dspy.context(lm=lm):
        cot = ChainOfThought(dspy.Signature(input_type=ProblemInput, output_type=SolutionOutput))
        result = await cot.acall(ProblemInput(problem="Is the sky blue?"))

        assert isinstance(result, SolutionOutput)
        assert result.solution == "Yes"
        assert result.reasoning == "Logical deduction"


def test_chain_of_thought_typed_signature_output_type_coercion():
    """Test that ChainOfThought with typed signatures returns properly typed instances."""

    @dataclass
    class InputData:
        query: str

    class OutputData:
        reasoning: str
        result: str

    lm = DummyLM([{"reasoning": "step-by-step analysis", "result": "conclusion"}])
    dspy.configure(lm=lm)

    cot = ChainOfThought(dspy.Signature(input_type=InputData, output_type=OutputData))
    result = cot(InputData(query="test"))

    # Verify the result is properly typed
    assert isinstance(result, OutputData)
    assert hasattr(result, "reasoning")
    assert hasattr(result, "result")
    assert result.reasoning == "step-by-step analysis"
    assert result.result == "conclusion"
