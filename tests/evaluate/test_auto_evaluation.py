import pytest

import dspy
from dspy.evaluate.auto_evaluation import CompleteAndGrounded, SemanticF1, f1_score
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


def judge_answers(metric_type, first, second):
    if metric_type is SemanticF1:
        return [{"plan": "Comparing responses", "precision": first, "recall": second}]
    return [
        {
            "plan": "Analyzing completeness",
            "ground_truth_key_ideas": "the answer is 2",
            "system_response_key_ideas": "the answer is 2",
            "discussion": "both match",
            "completeness": first,
        },
        {
            "plan": "Analyzing groundedness",
            "system_response_claims": "1+1=2",
            "discussion": "supported by context",
            "groundedness": second,
        },
    ]


@pytest.mark.parametrize("metric_type", [SemanticF1, CompleteAndGrounded])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("trace", [None, True])
def test_judge_rejects_nonfinite_components(metric_type, value, component, trace):
    scores = [1.0, 1.0]
    scores[component] = value
    example = dspy.Example(question="What is 1+1?", response="2")
    prediction = dspy.Prediction(response="2", context="1+1=2")

    with dspy.context(lm=DummyLM(judge_answers(metric_type, *scores))):
        with pytest.raises(ValueError, match="finite"):
            metric_type()(example, prediction, trace=trace)


@pytest.mark.parametrize("metric_type", [SemanticF1, CompleteAndGrounded])
def test_evaluate_counts_nonfinite_judgment_as_failure(metric_type):
    example = dspy.Example(question="What is 1+1?", response="2").with_inputs("question")

    def program(question):
        return dspy.Prediction(response="2", context="1+1=2")

    with dspy.context(lm=DummyLM(judge_answers(metric_type, "NaN", 1.0))):
        result = dspy.Evaluate(devset=[example], metric=metric_type(), num_threads=1, max_errors=2, failure_score=0.0)(
            program
        )

    assert result.score == 0.0
    assert result.results[0][2] == 0.0


@pytest.mark.parametrize(
    "precision,recall,expected",
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.8, 0.6, 0.6857142857142857),
        (-0.5, 1.0, 0.0),
        (1.0, -0.5, 0.0),
        (1.5, 1.0, 1.0),
        (1.0, 1.5, 1.0),
    ],
)
def test_f1_score_preserves_finite_scores_and_clamping(precision, recall, expected):
    assert f1_score(precision, recall) == pytest.approx(expected)


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
