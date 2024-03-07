import dspy
import typing as t
from dspy.signatures.signature import Signature, InputField, OutputField
from dspy.backends.lm.base import BaseLM, GeneratedOutput
from dspy.backends.template import TemplateBackend
from dspy.utils.dummies import DummyLanguageModel


class Emotion(Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""

    sentence = InputField()
    sentiment = OutputField()


class COTCheckCitationFaithfulness(Signature):
    """Verify that the text is based on the provided context."""

    context = InputField(desc="facts here are assumed to be true")
    text = InputField()
    rationale = OutputField(
        desc="Think step by step in order to generate the faithfulness.",
    )
    faithfulness = OutputField(
        desc="True/False indicating if text is faithful to context"
    )


def test_backend_complete_generation():
    # Initialize Backend
    dummy_lm = DummyLanguageModel(answers={1: ["Joy", "Joy", "Joy", "Joy", "Joy"]})
    backend = TemplateBackend(lm=dummy_lm)
    dspy.settings.configure(backend=backend)

    # Generate Sample Signature
    n = 5
    x = backend(Emotion, sentence="This is a positive sentence", n=n)
    assert len(x.sentence) == n
    assert len(x.sentiment) == n


def test_backend_with_recover():
    # Initialize Backend
    dummy_lm = DummyLanguageModel(
        answers={
            1: [
                "produce the faithfulness. We know that Lee has two loan spells in League One last term."
            ],
            2: ["True"],
        }
    )
    backend = TemplateBackend(lm=dummy_lm)
    dspy.settings.configure(backend=backend)

    # Generate Incomplete on the first try
    # Nothing should be returned from the generation as no results were complete
    n = 1
    x = backend(
        COTCheckCitationFaithfulness,
        context=["The 21-year-old made seven appearances for the Hammers."],
        text="Lee scored 3 goals for Colchester United.",
        recover=False,
    )

    assert len(x) == 0

    # Initialize Backend
    dummy_lm = DummyLanguageModel(
        answers={
            1: [
                "produce the faithfulness. We know that Lee has two loan spells in League One last term."
            ],
            2: ["True"],
        }
    )
    backend = TemplateBackend(lm=dummy_lm)
    dspy.settings.configure(backend=backend)

    # Generate Complete after recovery
    n = 1
    x = backend(
        COTCheckCitationFaithfulness,
        context=["The 21-year-old made seven appearances for the Hammers."],
        text="Lee scored 3 goals for Colchester United.",
        recover=True,
        n=n,
    )

    assert "rationale" in x
    assert "faithfulness" in x
