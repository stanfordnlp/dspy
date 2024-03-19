from dspy import Signature, ProgramOfThought
import dspy
from dspy.utils import DummyLM, DummyLanguageModel
import textwrap

from dspy.backends import TemplateBackend


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def test_pot_code_generation():
    pot = ProgramOfThought(BasicQA)

    lm = DummyLanguageModel(
        answers=[
            ["Reason_A\n\nCode: ```python\nresult = 1+1\n```"],
            ["Reason_B\n\nAnswer: 2"],
        ]
    )
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend, cache=False)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"


def test_pot_code_generation_with_error():
    pot = ProgramOfThought(BasicQA)

    lm = DummyLanguageModel(
        answers=[
            ["Reason_A\n\nCode:\n```python\nresult = 1+0/0\n```"],
            ["Reason_B\n\nCode:\n```python\nresult = 1+1\n```"],
            ["Reason_C\n\nAnswer: 2"],
        ]
    )
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend, cache=False)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
