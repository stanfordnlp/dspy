import pytest
import dspy
from dspy.signatures.signature import Signature, InputField, OutputField
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
    dummy_lm = DummyLanguageModel(answers=[["Joy", "Joy", "Joy", "Joy", "Joy"]])
    backend = TemplateBackend(lm=dummy_lm)
    with dspy.settings.context(backend=backend, lm=None):
        # Generate Sample Signature
        n = 5
        x = backend(Emotion, sentence="This is a positive sentence", n=n)
        assert len(x) == n
        assert x.examples[0].sentence == "This is a positive sentence"
        assert x.examples[0].sentiment == "Joy"


def test_backend_with_recover():
    # Initialize Backend
    dummy_lm = DummyLanguageModel(
        answers=[
            [
                "produce the faithfulness. We know that Lee has two loan spells in League One last term."
            ],
            ["True"],
        ],
    )
    backend = TemplateBackend(lm=dummy_lm)
    with dspy.settings.context(backend=backend, lm=None):
        # Generate Incomplete on the first try
        # Nothing should be returned from the generation as no results were complete
        n = 1
        with pytest.raises(Exception):
            backend(
                COTCheckCitationFaithfulness,
                context=["The 21-year-old made seven appearances for the Hammers."],
                text="Lee scored 3 goals for Colchester United.",
            )

    # Initialize Backend
    dummy_lm = DummyLanguageModel(
        answers=[
            [
                "produce the faithfulness. We know that Lee has two loan spells in League One last term."
            ],
            ["True"],
        ]
    )
    backend = TemplateBackend(lm=dummy_lm)
    with dspy.settings.context(backend=backend, lm=None, cache=False):

        # Generate Complete after recovery
        n = 1
        x = backend(
            COTCheckCitationFaithfulness,
            context=["The 21-year-old made seven appearances for the Hammers."],
            text="Lee scored 3 goals for Colchester United.",
            attempts=2,
            n=n,
        )

        assert x.examples[0].rationale is not None

        assert x.rationale
        assert x.faithfulness
