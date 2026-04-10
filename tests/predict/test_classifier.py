"""Tests for the Classifier module."""

import math
import unittest.mock as mock
from typing import Literal

import pytest

import dspy
from dspy import Classifier
from dspy.utils import DummyLM


# ---------------------------------------------------------------------------
# Shared signatures
# ---------------------------------------------------------------------------

class SentimentSignature(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()


class BinarySignature(dspy.Signature):
    """Binary yes/no classification."""

    question: str = dspy.InputField()
    answer: Literal["yes", "no"] = dspy.OutputField()


class PlainSignature(dspy.Signature):
    """A signature with no Literal-typed output field."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class MultiOutputSignature(dspy.Signature):
    """Signature with one Literal and one plain output field."""

    sentence: str = dspy.InputField()
    topic: str = dspy.OutputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()


class MultiLiteralSignature(dspy.Signature):
    """Signature with two Literal-typed output fields."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative"] = dspy.OutputField()
    topic: Literal["sports", "politics"] = dspy.OutputField()


# ===================================================================
# 1. Single prediction (default mode)
# ===================================================================

def test_single_prediction_default():
    """Default mode (need_confidence=False) runs a single prediction without confidence."""
    lm = DummyLM([{"sentiment": "positive"}])
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature)
    result = classifier(sentence="Great movie!")

    assert result.sentiment == "positive"
    assert not hasattr(result, "agreement_rate")
    assert not hasattr(result, "entropy")
    assert not hasattr(result, "prediction_counts")


def test_single_prediction_binary():
    """Single prediction works with binary signature."""
    lm = DummyLM([{"answer": "yes"}])
    dspy.configure(lm=lm)

    classifier = Classifier(BinarySignature)
    result = classifier(question="Is the sky blue?")
    assert result.answer == "yes"


# ===================================================================
# 2. Unanimous agreement
# ===================================================================

def test_unanimous_agreement():
    """All N samples return the same class -> agreement_rate=1.0, entropy=0.0."""
    answers = [{"sentiment": "positive"}] * 10
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Great movie!")

    assert result.sentiment == "positive"
    assert result.agreement_rate == 1.0
    assert result.entropy == 0.0
    assert result.prediction_counts == {"positive": 10}


def test_unanimous_negative():
    """Unanimous agreement on a non-default class."""
    answers = [{"sentiment": "negative"}] * 8
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=8)
    result = classifier(sentence="Terrible movie!")

    assert result.sentiment == "negative"
    assert result.agreement_rate == 1.0
    assert result.entropy == 0.0
    assert result.prediction_counts == {"negative": 8}


# ===================================================================
# 3. Mixed responses
# ===================================================================

def test_mixed_responses_majority():
    """7/10 positive, 3/10 negative -> majority positive, agreement_rate=0.7."""
    answers = [{"sentiment": "positive"}] * 7 + [{"sentiment": "negative"}] * 3
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Mostly good but some issues.")

    assert result.sentiment == "positive"
    assert result.agreement_rate == pytest.approx(0.7)
    assert 0.0 < result.entropy < 1.0
    assert result.prediction_counts == {"positive": 7, "negative": 3}


def test_three_way_split():
    """Three-way split: 5 positive, 3 negative, 2 neutral."""
    answers = (
        [{"sentiment": "positive"}] * 5
        + [{"sentiment": "negative"}] * 3
        + [{"sentiment": "neutral"}] * 2
    )
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Complex feelings.")

    assert result.sentiment == "positive"
    assert result.agreement_rate == pytest.approx(0.5)
    assert 0.0 < result.entropy < 1.0
    assert result.prediction_counts == {"positive": 5, "negative": 3, "neutral": 2}


# ===================================================================
# 4. Entropy calculations
# ===================================================================

def test_entropy_max_two_classes():
    """5/10 positive, 5/10 negative -> entropy == 1.0 (max for 2 classes)."""
    answers = [{"sentiment": "positive"}] * 5 + [{"sentiment": "negative"}] * 5
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Mixed feelings.")

    assert result.agreement_rate == pytest.approx(0.5)
    assert result.entropy == pytest.approx(1.0)


def test_entropy_max_three_classes():
    """Uniform 3-way split -> entropy close to 1.0."""
    answers = (
        [{"sentiment": "positive"}] * 4
        + [{"sentiment": "negative"}] * 3
        + [{"sentiment": "neutral"}] * 3
    )
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Eh.")

    # Not perfectly uniform, but entropy should be high
    assert result.entropy > 0.9


def test_entropy_perfect_uniform_three_classes():
    """Perfectly uniform 3-way split (9 samples, 3 each) -> entropy == 1.0."""
    answers = (
        [{"sentiment": "positive"}] * 3
        + [{"sentiment": "negative"}] * 3
        + [{"sentiment": "neutral"}] * 3
    )
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=9)
    result = classifier(sentence="Balanced.")

    assert result.entropy == pytest.approx(1.0)


# ===================================================================
# 5. Classification field detection
# ===================================================================

def test_auto_detect_literal_field():
    """Single Literal field is auto-detected."""
    classifier = Classifier(SentimentSignature)
    assert classifier.classification_field == "sentiment"


def test_auto_detect_with_mixed_outputs():
    """Literal field detected even when mixed with plain output fields."""
    classifier = Classifier(MultiOutputSignature)
    assert classifier.classification_field == "sentiment"


def test_explicit_matching_field_accepted():
    """Explicit classification_field matching the only Literal field is accepted."""
    classifier = Classifier(SentimentSignature, classification_field="sentiment")
    assert classifier.classification_field == "sentiment"


def test_explicit_mismatching_field_raises():
    """Explicit classification_field not matching the only Literal field raises."""
    with pytest.raises(ValueError, match="does not match the only Literal-typed output field"):
        Classifier(SentimentSignature, classification_field="wrong_field")


def test_no_literal_field_raises():
    """Signature without Literal-typed output field raises ValueError."""
    with pytest.raises(ValueError, match="requires at least one Literal-typed output field"):
        Classifier(PlainSignature)


def test_multiple_literal_no_field_raises():
    """Multiple Literal fields without explicit classification_field raises."""
    with pytest.raises(ValueError, match="multiple Literal-typed output fields"):
        Classifier(MultiLiteralSignature)


def test_multiple_literal_wrong_field_raises():
    """Explicit classification_field not in any Literal field raises."""
    with pytest.raises(ValueError, match="multiple Literal-typed output fields"):
        Classifier(MultiLiteralSignature, classification_field="wrong")


def test_multiple_literal_explicit_field():
    """Specifying classification_field among multiple Literal fields works."""
    answers = [{"sentiment": "positive", "topic": "sports"}] * 5
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(
        MultiLiteralSignature,
        need_confidence=True,
        num_samples=5,
        classification_field="topic",
    )
    assert classifier.classification_field == "topic"

    result = classifier(sentence="The team won!")
    assert result.topic == "sports"
    assert result.prediction_counts == {"sports": 5}


def test_multiple_literal_select_sentiment():
    """Can select the other Literal field when multiple exist."""
    answers = [{"sentiment": "positive", "topic": "sports"}] * 5
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(
        MultiLiteralSignature,
        need_confidence=True,
        num_samples=5,
        classification_field="sentiment",
    )
    assert classifier.classification_field == "sentiment"
    result = classifier(sentence="Great game!")
    assert result.prediction_counts == {"positive": 5}


# ===================================================================
# 6. num_samples handling
# ===================================================================

def test_custom_num_samples():
    """num_samples controls the number of predictions made."""
    answers = [{"sentiment": "positive"}] * 5
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=5)
    result = classifier(sentence="Good book!")

    assert sum(result.prediction_counts.values()) == 5


def test_num_samples_three():
    """Works with small num_samples."""
    answers = [{"sentiment": "positive"}] * 2 + [{"sentiment": "negative"}]
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=3)
    result = classifier(sentence="Ok-ish.")

    assert sum(result.prediction_counts.values()) == 3
    assert result.sentiment == "positive"
    assert result.agreement_rate == pytest.approx(2 / 3)


# ===================================================================
# 7. Temperature handling
# ===================================================================

def test_temperature_not_injected_when_absent():
    """When no temperature is set in call config, Classifier does not inject one."""
    captured_configs = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    lm = DummyLM([{"sentiment": "positive"}] * 10)
    dspy.configure(lm=lm)
    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)

    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!")

    assert len(captured_configs) == 10
    assert all(c.get("temperature") is None for c in captured_configs)
    assert all(c.get("n") == 1 for c in captured_configs)


def test_low_temperature_bumped():
    """Temperature < 0.15 is bumped to 0.5 for diverse sampling."""
    captured_configs = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    lm = DummyLM([{"sentiment": "positive"}] * 10)
    dspy.configure(lm=lm)
    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)

    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!", config={"temperature": 0.1})

    assert all(c.get("temperature") == pytest.approx(0.5) for c in captured_configs)


def test_high_temperature_preserved():
    """Temperature >= 0.15 is preserved as-is."""
    captured_configs = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    lm = DummyLM([{"sentiment": "positive"}] * 10)
    dspy.configure(lm=lm)
    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)

    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!", config={"temperature": 1.0})

    assert all(c.get("temperature") == pytest.approx(1.0) for c in captured_configs)


def test_temperature_at_boundary():
    """Temperature exactly at 0.15 is NOT bumped (boundary is strictly less-than)."""
    captured_configs = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    lm = DummyLM([{"sentiment": "positive"}] * 5)
    dspy.configure(lm=lm)
    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=5)

    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!", config={"temperature": 0.15})

    assert all(c.get("temperature") == pytest.approx(0.15) for c in captured_configs)


def test_zero_temperature_bumped():
    """Temperature 0.0 is bumped to 0.5."""
    captured_configs = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        captured_configs.append(dict(kwargs.get("config", {})))
        return original_forward(self, **kwargs)

    lm = DummyLM([{"sentiment": "positive"}] * 5)
    dspy.configure(lm=lm)
    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=5)

    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Nice!", config={"temperature": 0.0})

    assert all(c.get("temperature") == pytest.approx(0.5) for c in captured_configs)


# ===================================================================
# 8. Async forward
# ===================================================================

@pytest.mark.asyncio
async def test_async_single_prediction():
    """Async single prediction works like sync."""
    lm = DummyLM([{"sentiment": "positive"}])
    with dspy.context(lm=lm):
        classifier = Classifier(SentimentSignature)
        result = await classifier.acall(sentence="Great movie!")

    assert result.sentiment == "positive"
    assert not hasattr(result, "agreement_rate")


@pytest.mark.asyncio
async def test_async_with_confidence():
    """aforward produces the same structure as forward in confidence mode."""
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


# ===================================================================
# 9. String signatures
# ===================================================================

def test_string_signature():
    """Classifier works with string-based signatures containing Literal."""
    sig = dspy.Signature("text -> label: Literal['spam', 'ham']")
    answers = [{"label": "spam"}] * 3
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(sig, need_confidence=True, num_samples=3)
    result = classifier(text="Buy now! Limited offer!")

    assert result.label == "spam"
    assert result.prediction_counts == {"spam": 3}


# ===================================================================
# 10. classes property and __repr__
# ===================================================================

def test_classes_property():
    """The classes property returns the allowed labels."""
    classifier = Classifier(SentimentSignature)
    assert set(classifier.classes) == {"positive", "negative", "neutral"}


def test_classes_property_binary():
    """classes property works for binary classification."""
    classifier = Classifier(BinarySignature)
    assert set(classifier.classes) == {"yes", "no"}


def test_repr():
    """__repr__ includes key configuration."""
    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=20)
    r = repr(classifier)
    assert "sentiment" in r
    assert "need_confidence=True" in r
    assert "num_samples=20" in r


# ===================================================================
# 11. Edge cases
# ===================================================================

def test_single_sample_confidence():
    """num_samples=1 with confidence returns agreement_rate=1.0."""
    lm = DummyLM([{"sentiment": "neutral"}])
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=1)
    result = classifier(sentence="OK.")

    assert result.sentiment == "neutral"
    assert result.agreement_rate == 1.0
    assert result.entropy == 0.0
    assert result.prediction_counts == {"neutral": 1}


def test_prediction_counts_sum_equals_num_samples():
    """prediction_counts values always sum to num_samples."""
    answers = (
        [{"sentiment": "positive"}] * 6
        + [{"sentiment": "negative"}] * 3
        + [{"sentiment": "neutral"}] * 1
    )
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=10)
    result = classifier(sentence="Somewhat good.")

    assert sum(result.prediction_counts.values()) == 10


def test_rollout_ids_are_unique():
    """Each sample call gets a unique rollout_id in its config."""
    captured_ids = []
    original_forward = dspy.Predict.forward

    def capturing_forward(self, **kwargs):
        config = kwargs.get("config", {})
        if "rollout_id" in config:
            captured_ids.append(config["rollout_id"])
        return original_forward(self, **kwargs)

    lm = DummyLM([{"sentiment": "positive"}] * 5)
    dspy.configure(lm=lm)
    classifier = Classifier(SentimentSignature, need_confidence=True, num_samples=5)

    with mock.patch.object(dspy.Predict, "forward", capturing_forward):
        classifier(sentence="Test")

    assert len(captured_ids) == 5
    assert len(set(captured_ids)) == 5  # all unique
