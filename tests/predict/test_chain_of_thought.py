import dspy
from dspy import ChainOfThought
from dspy.utils import DummyLM


def test_initialization_with_string_signature():
    lm = DummyLM([{"reasoning": "find the number after 1", "answer": "2"}])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThought("question -> answer")
    assert list(predict.predict.signature.output_fields.keys()) == [
        "reasoning",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"


def test_cot_skips_with_reasoning_model():
    lm = DummyLM([{"answer": "2"}])
    lm.reasoning_model = True
    dspy.settings.configure(lm=lm)
    signature = dspy.Signature("question -> answer")
    predict = ChainOfThought(signature)
    assert list(predict.plain_predict.signature.output_fields.keys()) == [
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"
