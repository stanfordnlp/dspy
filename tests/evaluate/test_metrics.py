# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import pytest

import dspy
from dspy.evaluate.metrics import answer_exact_match
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
    "answer, pred_answer, expected",
    [
        (42, 42, True),
        (42, "42", True),
        ("42", 42, True),
        (3.5, 3.5, True),
        (True, True, True),
        ([42, "forty two"], 42, True),
        (42, 43, False),
    ],
)
def test_answer_exact_match_non_string_answers(answer, pred_answer, expected):
    example = dspy.Example(question="What is 6*7?", answer=answer).with_inputs("question")
    pred = dspy.Prediction(answer=pred_answer)
    assert answer_exact_match(example, pred) is expected
