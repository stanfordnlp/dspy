import dspy
from dspy.evaluate.auto_evaluation import (
    CompleteAndGrounded,
    SemanticF1,
    _count_mentioned_numbers,
    _count_numbered_items,
)
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


def test_semantic_f1_returns_prediction_without_trace():
    # Configure with a dummy LM that returns precision and recall values
    # ChainOfThought adds a "reasoning" field to the output
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Comparing the responses",
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
                    "reasoning": "Comparing the responses",
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
                    "reasoning": "Comparing the responses",
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
                    "reasoning": "Analyzing completeness",
                    "ground_truth_key_ideas": "the answer is 2",
                    "system_response_key_ideas": "the answer is 2",
                    "discussion": "both match",
                    "completeness": 1.0,
                },
                {
                    "reasoning": "Analyzing groundedness",
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
                    "reasoning": "Analyzing completeness",
                    "ground_truth_key_ideas": "the answer is 2",
                    "system_response_key_ideas": "the answer is 2",
                    "discussion": "both match",
                    "completeness": 0.9,
                },
                {
                    "reasoning": "Analyzing groundedness",
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
                    "reasoning": "Comparing first response",
                    "precision": 0.8,
                    "recall": 0.6,
                },
                {
                    "reasoning": "Comparing second response",
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


# --- Tests for computed SemanticF1 mode ---


def test_count_numbered_items():
    assert _count_numbered_items("1. first idea\n2. second idea\n3. third idea") == 3
    assert _count_numbered_items("1) first\n2) second") == 2
    assert _count_numbered_items("no numbered items here") == 0
    assert _count_numbered_items("") == 0


def test_count_mentioned_numbers():
    assert _count_mentioned_numbers("1, 3, 4") == 3
    assert _count_mentioned_numbers("1, 2") == 2
    assert _count_mentioned_numbers("None") == 0
    assert _count_mentioned_numbers("") == 0
    # Duplicate numbers should be counted once
    assert _count_mentioned_numbers("1, 1, 2") == 2


def test_semantic_f1_computed_mode():
    # 3 ground truth ideas, 2 covered -> recall = 2/3
    # 2 system response ideas, 2 grounded -> precision = 2/2 = 1.0
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Enumerating and matching ideas",
                    "ground_truth_key_ideas": "1. Earth orbits the Sun\n2. Earth has one moon\n3. Earth has water",
                    "system_response_key_ideas": "1. Earth orbits the Sun\n2. Earth has oceans",
                    "ground_truth_ideas_covered_by_system": "1, 3",
                    "system_response_ideas_grounded_in_truth": "1, 2",
                }
            ]
        )
    )

    example = dspy.Example(question="Tell me about Earth", response="ground truth")
    pred = dspy.Prediction(response="system response")

    metric = SemanticF1(decompositional="computed")
    result = metric(example, pred)

    assert isinstance(result, Prediction)
    # recall = 2/3, precision = 2/2 = 1.0
    expected_recall = 2 / 3
    expected_precision = 1.0
    expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
    assert abs(result.score - expected_f1) < 0.001


def test_semantic_f1_computed_mode_with_trace():
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Enumerating and matching ideas",
                    "ground_truth_key_ideas": "1. idea A\n2. idea B",
                    "system_response_key_ideas": "1. idea A\n2. idea B",
                    "ground_truth_ideas_covered_by_system": "1, 2",
                    "system_response_ideas_grounded_in_truth": "1, 2",
                }
            ]
        )
    )

    example = dspy.Example(question="test", response="answer")
    pred = dspy.Prediction(response="response")

    metric = SemanticF1(threshold=0.5, decompositional="computed")
    result = metric(example, pred, trace=True)

    assert isinstance(result, Prediction)
    assert isinstance(result.score, bool)
    assert result.score is True  # F1 = 1.0 > 0.5 threshold


def test_semantic_f1_computed_mode_no_matches():
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "No overlap found",
                    "ground_truth_key_ideas": "1. idea A\n2. idea B",
                    "system_response_key_ideas": "1. idea C",
                    "ground_truth_ideas_covered_by_system": "None",
                    "system_response_ideas_grounded_in_truth": "None",
                }
            ]
        )
    )

    example = dspy.Example(question="test", response="answer")
    pred = dspy.Prediction(response="response")

    metric = SemanticF1(decompositional="computed")
    result = metric(example, pred)

    assert isinstance(result, Prediction)
    assert result.score == 0.0


def test_semantic_f1_computed_mode_backward_compat():
    """Verify that existing decompositional=False and decompositional=True modes still work."""
    # Test decompositional=False (default)
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Comparing responses",
                    "precision": 0.9,
                    "recall": 0.8,
                }
            ]
        )
    )

    example = dspy.Example(question="test", response="answer")
    pred = dspy.Prediction(response="response")

    metric = SemanticF1(decompositional=False)
    result = metric(example, pred)
    expected_f1 = 2 * (0.9 * 0.8) / (0.9 + 0.8)
    assert abs(result.score - expected_f1) < 0.001

    # Test decompositional=True
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Detailed comparison",
                    "ground_truth_key_ideas": "key ideas",
                    "system_response_key_ideas": "key ideas",
                    "discussion": "they overlap",
                    "precision": 0.7,
                    "recall": 0.6,
                }
            ]
        )
    )

    metric = SemanticF1(decompositional=True)
    result = metric(example, pred)
    expected_f1 = 2 * (0.7 * 0.6) / (0.7 + 0.6)
    assert abs(result.score - expected_f1) < 0.001
