import dspy
from dspy import ChainOfDraft
from dspy.utils import DummyLM


def test_initialization_with_string_signature():
    lm = DummyLM([{"draft": "find the number after 1", "answer": "2"}])
    dspy.settings.configure(lm=lm)
    predict = ChainOfDraft("question -> answer")
    assert list(predict.predict.signature.output_fields.keys()) == [
        "draft",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"
