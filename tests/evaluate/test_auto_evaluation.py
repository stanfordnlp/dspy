import math

import pytest

import dspy
from dspy.evaluate.auto_evaluation import CompleteAndGrounded, SemanticF1, f1_score
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


def test_semantic_f1_returns_prediction_without_trace():
    # Configure with a dummy LM that returns precision and recall values
    # ChainOfThought requests a "plan" field from non-reasoning models.
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "plan": "Comparing the responses",
                    "precision": 1.0,
                    "recall": 1.0,
                }
            ]
        )
    )

    # Create example and prediction
    example = dspy.Example(question="What is 1+1?", response="2")
    pred = dspy.Prediction(response="2")

    # Test SemanticF1
    metric = SemanticF1()
    result = metric(example, pred)

    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, (int, float, bool))


def test_semantic_f1_returns_prediction_with_trace():
    # Configure with a dummy LM
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "plan": "Comparing the responses",
                    "precision": 1.0,
                    "recall": 1.0,
                }
            ]
        )
    )

    # Create example and prediction
    example = dspy.Example(question="What is 1+1?", response="2")
    pred = dspy.Prediction(response="2")

    # Test SemanticF1 with trace
    metric = SemanticF1(threshold=0.5)
    result = metric(example, pred, trace=True)

    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, bool)


def test_semantic_f1_score_value():
    # Configure with a dummy LM that returns specific precision and recall
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "plan": "Comparing the responses",
                    "precision": 0.8,
                    "recall": 0.6,
                }
            ]
        )
    )

    # Create example and prediction
    example = dspy.Example(question="test", response="answer")
    pred = dspy.Prediction(response="response")

    # Test SemanticF1
    metric = SemanticF1()
    result = metric(example, pred)

    expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
    assert isinstance(result, Prediction)
    assert abs(result.score - expected_f1) < 0.001


def test_complete_and_grounded_returns_prediction_without_trace():
    # Configure with a dummy LM
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "plan": "Analyzing completeness",
                    "ground_truth_key_ideas": "the answer is 2",
                    "system_response_key_ideas": "the answer is 2",
                    "discussion": "both match",
                    "completeness": 1.0,
                },
                {
                    "plan": "Analyzing groundedness",
                    "system_response_claims": "1+1=2",
                    "discussion": "supported by context",
                    "groundedness": 1.0,
                },
            ]
        )
    )

    # Create example and prediction with context
    example = dspy.Example(question="What is 1+1?", response="2")
    pred = dspy.Prediction(response="2", context="context")

    # Test CompleteAndGrounded
    metric = CompleteAndGrounded()
    result = metric(example, pred)

    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, (int, float, bool))


def test_complete_and_grounded_returns_prediction_with_trace():
    # Configure with a dummy LM
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "plan": "Analyzing completeness",
                    "ground_truth_key_ideas": "the answer is 2",
                    "system_response_key_ideas": "the answer is 2",
                    "discussion": "both match",
                    "completeness": 0.9,
                },
                {
                    "plan": "Analyzing groundedness",
                    "system_response_claims": "1+1=2",
                    "discussion": "supported by context",
                    "groundedness": 0.8,
                },
            ]
        )
    )

    # Create example and prediction with context
    example = dspy.Example(question="What is 1+1?", response="2")
    pred = dspy.Prediction(response="2", context="context")

    # Test CompleteAndGrounded with trace
    metric = CompleteAndGrounded(threshold=0.7)
    result = metric(example, pred, trace=True)

    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, bool)


def test_semantic_f1_prediction_can_be_compared():
    # Configure with a dummy LM
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "plan": "Comparing first response",
                    "precision": 0.8,
                    "recall": 0.6,
                },
                {
                    "plan": "Comparing second response",
                    "precision": 0.9,
                    "recall": 0.7,
                },
            ]
        )
    )

    metric = SemanticF1()

    # Create two predictions with different scores
    example1 = dspy.Example(question="test1", response="answer1")
    pred1 = dspy.Prediction(response="response1")
    result1 = metric(example1, pred1)

    example2 = dspy.Example(question="test2", response="answer2")
    pred2 = dspy.Prediction(response="response2")
    result2 = metric(example2, pred2)

    assert isinstance(result1, Prediction)
    assert isinstance(result2, Prediction)
    assert result2.score > result1.score

class TestF1ScoreNonFiniteValues:
    """f1_score must reject non-finite components (NaN, +inf, -inf) instead of
    silently clamping them to valid scores.  See GitHub issue #10739."""

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_precision_raises(self, bad_value):
        with pytest.raises(ValueError, match="precision must be finite"):
            f1_score(bad_value, 1.0)

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_recall_raises(self, bad_value):
        with pytest.raises(ValueError, match="recall must be finite"):
            f1_score(1.0, bad_value)

    def test_both_non_finite_raises(self):
        with pytest.raises(ValueError):
            f1_score(float("nan"), float("nan"))

    def test_finite_values_still_work(self):
        """Ensure normal finite values are unaffected by the guard."""
        assert f1_score(1.0, 1.0) == 1.0
        assert f1_score(0.0, 0.0) == 0.0
        expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(f1_score(0.8, 0.6) - expected) < 1e-9

    def test_clamping_still_works_for_finite_out_of_range(self):
        """Values outside [0, 1] but finite should still be clamped."""
        expected = 2 * (1.0 * 0.5) / (1.0 + 0.5)
        assert abs(f1_score(1.5, 0.5) - expected) < 1e-9
        assert f1_score(-0.5, 1.0) == 0.0


class TestSemanticF1NanPropagation:
    """SemanticF1 must surface the ValueError from f1_score when the judge
    returns non-finite components."""

    def test_nan_precision_raises_in_metric(self):
        dspy.configure(
            lm=DummyLM(
                [{"plan": "fixture", "precision": "NaN", "recall": 1.0}]
            )
        )
        example = dspy.Example(question="q", response="a")
        pred = dspy.Prediction(response="b")
        with pytest.raises(ValueError, match="precision must be finite"):
            SemanticF1()(example, pred)

    def test_nan_recall_raises_in_metric(self):
        dspy.configure(
            lm=DummyLM(
                [{"plan": "fixture", "precision": 1.0, "recall": "NaN"}]
            )
        )
        example = dspy.Example(question="q", response="a")
        pred = dspy.Prediction(response="b")
        with pytest.raises(ValueError, match="recall must be finite"):
            SemanticF1()(example, pred)
