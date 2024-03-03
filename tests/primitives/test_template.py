import pytest
from dspy import Example, Signature, InputField, OutputField
from dspy.primitives import Template


class Emotion(Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""

    sentence = InputField()
    sentiment = OutputField()


EXAMPLE_TEMPLATE_PROMPTS = [
    (
        Emotion,
        [],
        "This is a positive test sentence.",
        "Joy",
        "Joy",
        "Classify emotion among sadness, joy, love, anger, fear, surprise.\n\n---\n\nFollow the following format.\n\nSentence: ${sentence}\nSentiment: ${sentiment}\n\n---\n\nSentence: This is a positive test sentence.\nSentiment:",
    )
]


def test_example_initialization():
    for signature, demos, sentence, _, _, prompt in EXAMPLE_TEMPLATE_PROMPTS:
        template = Template(signature)
        example = Example(sentence=sentence, demos=demos)
        assert template(example) == prompt


def test_template_extraction():
    for (
        signature,
        demos,
        sentence,
        sentiment,
        output,
        _,
    ) in EXAMPLE_TEMPLATE_PROMPTS:
        template = Template(signature)
        example = Example(sentence=sentence, demos=demos)
        assert template.extract(example, output) == Example(
            sentence=sentence, demos=demos, sentiment=sentiment
        )
