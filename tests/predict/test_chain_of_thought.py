import textwrap
import dspy
from dspy import ChainOfThought
from dspy.utils import DummyLM
from dspy.utils.dummies import DummyLanguageModel
from dspy.backends import TemplateBackend


def test_initialization_with_string_signature():
    lm = DummyLanguageModel(answers=[["find the number after 1\n\nAnswer: 2"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend)
    predict = ChainOfThought("question -> answer")
    assert list(predict.extended_signature.output_fields.keys()) == [
        "rationale",
        "answer",
    ]
    output = predict(question="What is 1+1?")
    assert output.answer == "2"
    assert output.rationale == "find the number after 1"
