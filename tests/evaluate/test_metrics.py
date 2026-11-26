# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import pytest

import dspy
from dspy.evaluate.metrics import answer_exact_match, answer_passage_match
from dspy.predict import Predict


def test_answer_exact_match_string():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_list():
    example = dspy.Example(
        question="What is 1+1?",
        answer=["2", "two"],
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_no_match():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["Paris", ""], False),
        (["Paris", "the"], False),
        (["Paris", "A"], False),
        (["Paris", "!"], False),
        (["", "an", "!"], False),
        (["", "Tokyo"], True),
    ],
)
def test_answer_passage_match_ignores_empty_normalized_answers(answers, expected):
    example = dspy.Example(answer=answers)
    pred = dspy.Prediction(context=["Tokyo is in Japan."])
    assert answer_passage_match(example, pred) is expected
