import textwrap

import dspy
from dspy import ChainOfThoughtWithHint
from dspy.utils import DummyLM


def test_cot_with_no_hint():
    lm = DummyLM([{"rationale": "find the number after 1", "answer": "2"}])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThoughtWithHint("question -> answer")
    # Check output fields have the right order
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"


def test_cot_with_hint():
    lm = DummyLM([{"rationale": "find the number after 1", "hint": "Is it helicopter?", "answer": "2"}])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThoughtWithHint("question -> answer")
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?", hint="think small").answer == "2"
