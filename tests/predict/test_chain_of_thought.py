import pytest

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
