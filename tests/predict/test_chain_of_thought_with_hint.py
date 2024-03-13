import dspy
from dspy import ChainOfThoughtWithHint
from dspy.utils import DummyLanguageModel
from dspy.backends import TemplateBackend


def test_cot_with_no_hint():
    lm = DummyLanguageModel(answers=[["find the number after 1\n\nAnswer: 2"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend)
    predict = ChainOfThoughtWithHint("question -> answer")
    # Check output fields have the right order
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"


def test_cot_with_hint():
    lm = DummyLanguageModel(answers=[["find the number after 1\n\nAnswer: 2"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend)
    predict = ChainOfThoughtWithHint("question -> answer")
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?", hint="think small").answer == "2"
