import dsp
from dspy import ChainOfThought
from dspy.utils import DummyLM

def test_initialization_with_string_signature():
    dsp.settings.lm = DummyLM(["find the number after 1", "2"])
    predict = ChainOfThought("question -> answer")
    assert list(predict.extended_signature.output_fields.keys()) == ["rationale", "answer"]
    assert predict(question="What is 1+1?").answer == "2"