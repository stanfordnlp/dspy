# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import dspy
from dspy.evaluate.metrics import (
    EM,
    F1,
    HotPotF1,
    answer_exact_match,
    answer_passage_match,
)
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


def test_answer_exact_match_tuple_and_set():
    example_tuple = dspy.Example(
        question="What is 1+1?",
        answer=("2", "two"),
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example_tuple, pred)

    example_set = dspy.Example(
        question="What is 1+1?",
        answer={"2", "two"},
    ).with_inputs("question")
    assert answer_exact_match(example_set, pred)


def test_answer_exact_match_no_match():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)


def test_answer_passage_match_string_context():
    example = dspy.Example(answer="Eiffel Tower")
    pred = dspy.Prediction(context="The Eiffel Tower is located in Paris, France.")
    assert answer_passage_match(example, pred)

    pred_no_match = dspy.Prediction(context="The Louvre is in Paris.")
    assert not answer_passage_match(example, pred_no_match)


def test_answer_passage_match_list_and_tuple_context():
    example = dspy.Example(answer=["Eiffel Tower", "Louvre"])
    pred_list = dspy.Prediction(context=["The Eiffel Tower is in Paris.", "Some other passage."])
    assert answer_passage_match(example, pred_list)

    example_tuple = dspy.Example(answer=("Eiffel Tower", "Louvre"))
    pred_tuple = dspy.Prediction(context=("The Eiffel Tower is in Paris.",))
    assert answer_passage_match(example_tuple, pred_tuple)


def test_answer_passage_match_none_or_missing_context():
    example = dspy.Example(answer="Eiffel Tower")
    assert not answer_passage_match(example, dspy.Prediction(context=None))
    assert not answer_passage_match(example, dspy.Prediction(context=[]))
    assert not answer_passage_match(example, dspy.Prediction(other="something"))


def test_em_f1_hotpot_empty_answers():
    assert not EM("paris", [])
    assert F1("paris", []) == 0.0
    assert HotPotF1("paris", []) == 0.0


def test_em_f1_hotpot_tuple_and_set():
    assert EM("paris", ("Paris", "France"))
    assert EM("paris", {"Paris", "France"})
    assert F1("paris", ("Paris",)) == 1.0
    assert HotPotF1("yes", ("yes", "no")) == 1.0


def test_answer_exact_match_empty_references():
    pred = dspy.Prediction(answer="Paris")
    for empty_ref in ([], (), set()):
        example = dspy.Example(answer=empty_ref)
        assert not answer_exact_match(example, pred, frac=1.0)
        assert not answer_exact_match(example, pred, frac=0.5)
        assert not answer_exact_match(example, pred, frac=0.0)
