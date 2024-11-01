import pytest
import dspy
from dspy import ChainOfThoughtWithHint
from dspy.utils import DSPDummyLM
from dspy.utils.dummies import DummyLM


def test_cot_with_no_hint():
    lm = DSPDummyLM(["find the number after 1", "2"])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThoughtWithHint("question -> answer")
    # Check output fields have the right order
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"

    final_convo = lm.get_convo(-1)
    assert final_convo.endswith(
        "Question: What is 1+1?\n"
        "Reasoning: Let's think step by step in order to find the number after 1\n"
        "Answer: 2"
    )


@pytest.mark.asyncio
async def test_async_cot_with_no_hint():
    lm = DummyLM([{"answer": "2"}])
    dspy.settings.configure(lm=lm, async_mode=True)
    predict = ChainOfThoughtWithHint("question -> answer")
    result = await predict(question="What is 1+1?")
    assert result.answer == "2"


def test_cot_with_hint():
    lm = DSPDummyLM(["find the number after 1", "2"])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThoughtWithHint("question -> answer")
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?", hint="think small").answer == "2"

    final_convo = lm.get_convo(-1)
    assert final_convo.endswith(
        "Question: What is 1+1?\n\n"
        "Reasoning: Let's think step by step in order to find the number after 1\n\n"
        "Hint: think small\n\n"
        "Answer: 2"
    )


@pytest.mark.asyncio
async def test_async_cot_with_hint():
    lm = DummyLM([{"answer": "2"}])
    dspy.settings.configure(lm=lm, async_mode=True)
    predict = ChainOfThoughtWithHint("question -> answer")
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    result = await predict(question="What is 1+1?", hint="think small")
    assert result.answer == "2"
