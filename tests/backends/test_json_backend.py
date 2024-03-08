import dspy
from dspy.utils.dummies import DummyLanguageModel
from dspy.backends.json import JSONBackend
from dspy.signatures.signature import Signature, InputField, OutputField


class Emotion(Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""

    sentence = InputField()
    sentiment = OutputField()


def test_backend_complete_generation():
    # Initialize Backend
    dummy_lm = DummyLanguageModel(
        answers={
            1: [
                """{"sentence": "This is a positive sentence", "sentiment": "Joy"}""",
                """{"sentence": "This is a positive sentence", "sentiment": "Joy"}""",
                """{"sentence": "This is a positive sentence", "sentiment": "Joy"}""",
                """{"sentence": "This is a positive sentence", "sentiment": "Joy"}""",
                """{"sentence": "This is a positive sentence", "sentiment": "Joy"}""",
            ]
        }
    )
    backend = JSONBackend(lm=dummy_lm)
    dspy.configure(backend=backend)

    # Generate Sample Signature
    n = 5
    x = backend(Emotion, sentence="This is a positive sentence", n=n)
    assert len(x) == n
    assert x.sentiment == "Joy"
