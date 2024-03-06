import dspy
import typing as t
from dspy.signatures.signature import Signature, InputField, OutputField
from dspy.backends.lm.base import BaseLM, GeneratedOutput
from dspy.backends.template import TemplateBackend


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


class DummyMessage:
    def __init__(self, content: str):
        self.content = content


class DummyChoice:
    def __init__(self, message: str):
        self.message = message


class DummyLanguageModel(BaseLM):
    answers: dict[int, list[str]]
    step: int = 1

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> t.List[GeneratedOutput]:
        if len(self.answers) > 1:
            self.step += 1
        return [
            (DummyChoice(DummyMessage(content))) for content in self.answers[self.step]
        ]

    def count_tokens(self, prompt: str) -> int:
        return len(prompt)


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
