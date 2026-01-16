# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import dspy
from dspy.evaluate.metrics import answer_exact_match, Recall, Precision
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
    assert not answer_exact_match(example, pred)


def test_precision_metric():
    # Full overlap
    assert Precision("approximate answer", ["approximate answer"]) == 1.0
    # Partial overlap
    # "approximate" is 1 token. "approximate answer" is 2 tokens. intersection is 1.
    # Precision = 1 / 1 = 1.0 ?? Wait.
    # prediction tokens: ["approximate"] (len 1)
    # ground truth: ["approximate", "answer"] (len 2)
    # intersection: 1.
    # Precision = 1/1 = 1.0. Correct.
    assert Precision("approximate", ["approximate answer"]) == 1.0

    # "answer" (1 token). "approximate answer" (2 tokens).
    # Precision = 1/1 = 1.0.
    assert Precision("answer", ["approximate answer"]) == 1.0

    # Prediction: "approximate answer extra" (3 tokens)
    # GT: "approximate answer" (2 tokens)
    # Intersection: 2.
    # Precision = 2 / 3 = 0.666...
    assert abs(Precision("approximate answer extra", ["approximate answer"]) - 0.666) < 0.01

    # No overlap
    assert Precision("wrong", ["approximate answer"]) == 0.0


def test_recall_metric():
    # Full overlap
    assert Recall("approximate answer", ["approximate answer"]) == 1.0

    # Partial overlap
    # Prediction: "approximate" (1 token)
    # GT: "approximate answer" (2 tokens)
    # Intersection: 1.
    # Recall = 1 / 2 = 0.5.
    assert Recall("approximate", ["approximate answer"]) == 0.5

    # Prediction: "approximate answer extra" (3 tokens)
    # GT: "approximate answer" (2 tokens)
    # Intersection: 2.
    # Recall = 2 / 2 = 1.0.
    assert Recall("approximate answer extra", ["approximate answer"]) == 1.0

    # No overlap
    assert Recall("wrong", ["approximate answer"]) == 0.0
