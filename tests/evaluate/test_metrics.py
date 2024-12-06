# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import dsp, dspy
from dspy.evaluate.metrics import (answer_exact_match,
                                   validate_answer_with_spBLEU,
                                   validate_answer_with_cosine_similarity
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

def test_answer_exact_match_no_match():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)


def test_validate_answer_with_spBLEU_match():
    example = dspy.Example(
        question="Translate 'hello' to Spanish.",
        answer="hola",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "hola"
    assert validate_answer_with_spBLEU(example, pred, threshold=50)


def test_validate_answer_with_spBLEU_no_match():
    example = dspy.Example(
        question="Translate 'hello' to Spanish.",
        answer="hola",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "adios"
    assert not validate_answer_with_spBLEU(example, pred, threshold=50)


def test_validate_answer_with_cosine_similarity_match():
    example = dspy.Example(
        question="Reformulate the text.",
        answer="The cat sat on the mat.",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "A cat is sitting on the mat."
    assert validate_answer_with_cosine_similarity(example, pred, threshold=0.85)


def test_validate_answer_with_cosine_similarity_no_match():
    example = dspy.Example(
        question="Reformulate the text.",
        answer="The cat sat on the mat.",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "The president is visiting Paris. It is a beautiful city."
    assert not validate_answer_with_cosine_similarity(example, pred, threshold=0.85)
