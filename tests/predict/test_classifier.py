import math
from typing import Literal

import pytest

import dspy
from dspy import Classifier
from dspy.utils import DummyLM


class SentimentSignature(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()


class PlainSignature(dspy.Signature):
    """Answer a question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class MultiOutputSignature(dspy.Signature):
    """Classify a sentence with a topic and sentiment."""
    sentence: str = dspy.InputField()
    topic: str = dspy.OutputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()


def test_single_prediction_default():
    """Default (need_confidence=False) runs a single prediction with no confidence fields."""
    lm = DummyLM([{"sentiment": "positive"}])
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature)
    result = classifier(sentence="Great movie!")

    assert result.sentiment == "positive"
    assert not hasattr(result, "agreement_rate")
    assert not hasattr(result, "entropy")
    assert not hasattr(result, "prediction_counts")


def test_unanimous_agreement():
    """All N samples return the same class → agreement_rate=1.0, entropy=0.0."""
    answers = [{"sentiment": "positive"}] * 10
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Great movie!")

    assert result.sentiment == "positive"
    assert result.agreement_rate == 1.0
    assert result.entropy == 0.0
    assert result.prediction_counts == {"positive": 10}


def test_mixed_responses():
    """7/10 positive, 3/10 negative → majority positive, agreement_rate=0.7."""
    answers = [{"sentiment": "positive"}] * 7 + [{"sentiment": "negative"}] * 3
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Mostly good but some issues.")

    assert result.sentiment == "positive"
    assert result.agreement_rate == pytest.approx(0.7)
    assert 0.0 < result.entropy < 1.0
    assert result.prediction_counts == {"positive": 7, "negative": 3}


def test_entropy_max_disagreement():
    """5/10 positive, 5/10 negative → entropy == 1.0 (max for 2 classes)."""
    answers = [{"sentiment": "positive"}] * 5 + [{"sentiment": "negative"}] * 5
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Mixed feelings.")

    assert result.agreement_rate == pytest.approx(0.5)
    assert result.entropy == pytest.approx(1.0)


def test_auto_detect_literal_field():
    """Signature with Literal field is auto-detected as the classification field."""
    lm = DummyLM([{"sentiment": "neutral"}] * 5)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, num_samples=5)
    assert classifier.classification_field == "sentiment"


def test_single_literal_explicit_matching_field():
    """Explicit classification_field matching the only Literal field is accepted."""
    lm = DummyLM([{"sentiment": "neutral"}] * 5)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, num_samples=5, classification_field="sentiment")
    assert classifier.classification_field == "sentiment"


def test_single_literal_mismatching_field_raises():
    """Explicit classification_field that doesn't match the only Literal field raises ValueError."""
    with pytest.raises(ValueError, match="does not match the only Literal-typed output field"):
        Classifier(SentimentSignature, num_samples=5, classification_field="wrong_field")


def test_explicit_classification_field_multi_literal():
    """User-specified classification_field selects among multiple Literal fields."""
    class MultiLiteralSignature(dspy.Signature):
        """Classify both sentiment and topic."""
        sentence: str = dspy.InputField()
        sentiment: Literal["positive", "negative"] = dspy.OutputField()
        topic: Literal["sports", "politics"] = dspy.OutputField()

    lm = DummyLM([{"sentiment": "positive", "topic": "sports"}] * 5)
    dspy.configure(lm=lm)

    classifier = Classifier(MultiLiteralSignature, need_confidence=True, num_samples=5, classification_field="topic")
    assert classifier.classification_field == "topic"

    result = classifier(sentence="The team won!")
    assert result.topic == "sports"
    assert result.prediction_counts == {"sports": 5}


def test_multiple_literal_fields_wrong_field_raises():
    """Explicit classification_field not in multiple Literal fields raises ValueError."""
    class MultiLiteralSignature(dspy.Signature):
        """Classify both sentiment and topic."""
        sentence: str = dspy.InputField()
        sentiment: Literal["positive", "negative"] = dspy.OutputField()
        topic: Literal["sports", "politics"] = dspy.OutputField()

    with pytest.raises(ValueError, match="multiple Literal-typed output fields"):
        Classifier(MultiLiteralSignature, num_samples=5, classification_field="wrong_field")


def test_no_literal_field_raises():
    """When no Literal field exists, raise ValueError."""
    with pytest.raises(ValueError, match="requires at least one Literal-typed output field"):
        Classifier(PlainSignature, num_samples=5)


def test_multiple_literal_fields_raises():
    """When multiple Literal fields exist without explicit classification_field, raise ValueError."""
    class MultiLiteralSignature(dspy.Signature):
        """Classify both sentiment and topic."""
        sentence: str = dspy.InputField()
        sentiment: Literal["positive", "negative"] = dspy.OutputField()
        topic: Literal["sports", "politics"] = dspy.OutputField()

    with pytest.raises(ValueError, match="multiple Literal-typed output fields"):
        Classifier(MultiLiteralSignature, num_samples=5)


def test_multiple_literal_fields_with_explicit_field():
    """When multiple Literal fields exist, specifying classification_field works."""
    class MultiLiteralSignature(dspy.Signature):
        """Classify both sentiment and topic."""
        sentence: str = dspy.InputField()
        sentiment: Literal["positive", "negative"] = dspy.OutputField()
        topic: Literal["sports", "politics"] = dspy.OutputField()

    lm = DummyLM([{"sentiment": "positive", "topic": "sports"}] * 5)
    dspy.configure(lm=lm)

    classifier = Classifier(MultiLiteralSignature, num_samples=5, classification_field="sentiment")
    assert classifier.classification_field == "sentiment"


def test_custom_num_samples():
    """Verify num_samples controls the number of completions requested."""
    answers = [{"sentiment": "positive"}] * 5
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=5)
    result = classifier(sentence="Good book!")

    assert sum(result.prediction_counts.values()) == 5


def test_temperature_set_when_absent():
    """When no temperature in call config, Classifier sets minimum 0.7 for diversity."""
    captured_configs = []

    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    lm = DummyLM([{"sentiment": "positive"}] * 10)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)

    import unittest.mock as mock
    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!")

    assert len(captured_configs) == 10
    assert all(c.get("temperature") == pytest.approx(0.7) for c in captured_configs)
    # Classifier explicitly sets n=1 to ensure compatibility with models that don't support n > 1
    assert all(c.get("n") == 1 for c in captured_configs)


def test_temperature_bumped_from_low():
    """Temperature < 0.7 in call config is bumped to 0.7."""
    lm = DummyLM([{"sentiment": "positive"}] * 10)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)

    captured_configs = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!", config={"temperature": 0.15})

    assert all(c.get("temperature") == pytest.approx(0.7) for c in captured_configs)


def test_high_temperature_preserved():
    """Temperature >= 0.7 in call config is preserved."""
    lm = DummyLM([{"sentiment": "positive"}] * 10)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)

    captured_configs = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!", config={"temperature": 1.0})

    assert all(c.get("temperature") == pytest.approx(1.0) for c in captured_configs)


@pytest.mark.asyncio
async def test_async_forward():
    """aforward produces the same structure as forward."""
    answers = [{"sentiment": "positive"}] * 8 + [{"sentiment": "negative"}] * 2
    lm = DummyLM(answers)
    with dspy.context(lm=lm):
        classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
        result = await classifier.acall(sentence="Really enjoyed it!")

    assert result.sentiment == "positive"
    assert result.agreement_rate == pytest.approx(0.8)
    assert hasattr(result, "entropy")
    assert hasattr(result, "prediction_counts")
    assert result.prediction_counts == {"positive": 8, "negative": 2}
