from dspy.primitives.prediction import (
    Completions,
)
from dspy.primitives.example import Example
from dspy.signatures.signature import Signature, InputField, OutputField
from dspy.primitives.prompt import Prompt


class QuestionSignature(Signature):
    """Provide the answer given a particular question"""

    question = InputField()
    answer = OutputField()


class COTSignature(Signature):
    """Provide the answer and rationale given a particular question"""

    question = InputField()
    rationale = OutputField()
    answer = OutputField()


def test_get_completions():
    examples = [
        Example(question="What is the first letter of the alphabet?", answer="a"),
        Example(question="What is the first letter of the alphabet?", answer="b"),
        Example(question="What is the first letter of the alphabet?", answer="a"),
    ]

    completions = Completions.new(
        signature=QuestionSignature,
        examples=examples,
        prompt=Prompt.from_str("DUMMY PROMPT"),
        kwargs={},
    )

    assert completions.question == "What is the first letter of the alphabet?"
    assert completions.answer == "a"
    assert completions[0].question == "What is the first letter of the alphabet?"
    assert completions[1].answer == "b"

    assert len(completions) == 3

    for example in completions:
        assert example


def test_completions_complete_checks():
    examples = [
        Example(question="What is the first letter of the alphabet?", answer="a"),
        Example(question="What is the first letter of the alphabet?"),
        Example(question="What is the first letter of the alphabet?", answer="a"),
    ]

    completions = Completions.new(
        signature=QuestionSignature,
        examples=examples,
        prompt=Prompt.from_str("DUMMY PROMPT"),
        kwargs={},
    )

    assert completions.has_complete_example()
    assert len(completions) == 3

    assert completions.get_farthest_example() == examples[0]
    completions.remove_incomplete()

    assert len(completions) == 2


def test_completions_numerous_answers():
    examples = [
        Example(question="What is the first letter of the alphabet?"),
        Example(question="What is the first letter of the alphabet?"),
        Example(
            question="What is the first letter of the alphabet?",
            rationale="the first letter of the alphabet",
            answer="a",
        ),
    ]

    completions = Completions.new(
        signature=QuestionSignature,
        examples=examples,
        prompt=Prompt.from_str("DUMMY PROMPT"),
        kwargs={},
    )

    assert len(completions) == 3
    completions.remove_incomplete()
    assert completions["rationale"] == "the first letter of the alphabet"
    assert completions.rationale == "the first letter of the alphabet"
    assert completions[0] == Example(
        question="What is the first letter of the alphabet?",
        rationale="the first letter of the alphabet",
        answer="a",
    )
