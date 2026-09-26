# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import dspy
from dspy.evaluate.metrics import EM, answer_exact_match, em_score
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


def test_em_score_does_not_match_on_degenerate_empty_normalization():
    assert em_score("", "the") is False
    assert em_score("", "a") is False
    assert em_score("", "--") is False
    assert em_score(" ", "the") is False
    assert em_score("...", "the") is False
    assert em_score("", "") is False
    assert em_score("the", "") is False


def test_em_score_matches_non_degenerate_strings():
    assert em_score("Paris", "paris") is True
    assert em_score("The Eiffel Tower", "Eiffel Tower") is True
    assert em_score("Paris", "Paris, France") is False


def test_em_multi_answer_list_ignores_stopword_placeholder_gold():
    assert EM("", ["Eiffel Tower", "the"]) is False
    assert EM("Eiffel Tower", ["Eiffel Tower", "the"]) is True
