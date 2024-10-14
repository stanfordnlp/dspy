import textwrap

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def test_pot_code_generation():
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.settings.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"


def test_pot_code_generation_with_error():
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+0/0\n```"},
            {"reasoning": "Reason_B", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_C", "answer": "2"},
        ]
    )
    dspy.settings.configure(lm=lm)

    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
