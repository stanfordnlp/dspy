import pytest

import dspy
from dspy.evaluate.auto_evaluation import CompleteAndGrounded, RAGGroundedRefusal, SemanticF1
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


def test_rag_grounded_refusal_correct_refusal_on_unanswerable():
    # No LM call is needed for the refusal branches
    example = dspy.Example(question="Who?", response="not enough information", answerable=False)
    pred = dspy.Prediction(response="not enough information", context="irrelevant text", refused=True)

    metric = RAGGroundedRefusal()
    result = metric(example, pred)

    assert result.score == 1.0
    assert "correctly refused" in result.feedback


def test_rag_grounded_refusal_answered_when_unanswerable():
    example = dspy.Example(question="Who?", response="not enough information", answerable=False)
    pred = dspy.Prediction(response="Paris", context="irrelevant text", refused=False)

    metric = RAGGroundedRefusal()
    result = metric(example, pred)

    assert result.score == 0.0
    assert "should refuse" in result.feedback


def test_rag_grounded_refusal_refused_when_answerable():
    example = dspy.Example(question="What is 1+1?", response="2", answerable=True)
    pred = dspy.Prediction(response="not enough information", context="1+1 equals 2", refused=True)

    metric = RAGGroundedRefusal()
    result = metric(example, pred)

    assert result.score == 0.0
    assert "refused although" in result.feedback


def test_rag_grounded_refusal_scores_answerable_answer():
    # Configure with a dummy LM: first call scores correctness, second scores groundedness
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Comparing the responses",
                    "precision": 0.8,
                    "recall": 0.6,
                },
                {
                    "reasoning": "Checking the claims",
                    "system_response_claims": "1+1=2",
                    "discussion": "supported by context",
                    "groundedness": 1.0,
                },
            ]
        )
    )

    example = dspy.Example(question="What is 1+1?", response="2", answerable=True)
    pred = dspy.Prediction(response="2", context="1+1 equals 2", refused=False)

    metric = RAGGroundedRefusal()
    result = metric(example, pred)

    correctness = 2 * (0.8 * 0.6) / (0.8 + 0.6)
    expected = 2 * (correctness * 1.0) / (correctness + 1.0)
    assert abs(result.score - expected) < 0.001
    assert "correct and grounded" in result.feedback


def test_rag_grounded_refusal_feedback_names_ungrounded_claims():
    # Correct answer, but the groundedness judge finds unsupported claims
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Comparing the responses",
                    "precision": 1.0,
                    "recall": 1.0,
                },
                {
                    "reasoning": "Checking the claims",
                    "system_response_claims": "several",
                    "discussion": "not supported",
                    "groundedness": 0.2,
                },
            ]
        )
    )

    example = dspy.Example(question="What is 1+1?", response="2", answerable=True)
    pred = dspy.Prediction(response="2, and the sky is green", context="1+1 equals 2", refused=False)

    metric = RAGGroundedRefusal()
    result = metric(example, pred)

    assert result.score < 0.66
    assert "not supported by the retrieved context" in result.feedback


def test_rag_grounded_refusal_with_trace_returns_bool():
    example = dspy.Example(question="Who?", response="not enough information", answerable=False)
    pred = dspy.Prediction(response="not enough information", context="irrelevant text", refused=True)

    metric = RAGGroundedRefusal(threshold=0.7)
    result = metric(example, pred, trace=True)

    assert isinstance(result.score, bool)
    assert result.score is True


def test_rag_grounded_refusal_is_refusal_callable():
    example = dspy.Example(question="Who?", response="not enough information", answerable=False)
    pred = dspy.Prediction(response="not enough information in the context", context="irrelevant text")

    metric = RAGGroundedRefusal(is_refusal=lambda response: "not enough information" in response)
    result = metric(example, pred)

    assert result.score == 1.0


def test_rag_grounded_refusal_requires_refusal_signal():
    example = dspy.Example(question="Who?", response="2", answerable=True)
    pred = dspy.Prediction(response="2", context="context")

    metric = RAGGroundedRefusal()
    with pytest.raises(ValueError, match="refusal signal"):
        metric(example, pred)


def test_rag_grounded_refusal_requires_answerable_field():
    example = dspy.Example(question="Who?", response="2")
    pred = dspy.Prediction(response="2", context="context", refused=False)

    metric = RAGGroundedRefusal()
    with pytest.raises(ValueError, match="answerable"):
        metric(example, pred)
