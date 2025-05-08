import pytest
from unittest.mock import patch
import dspy
from dspy.predict.refine import Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM
from dspy.signatures import Signature, InputField, OutputField

class SimpleQA(Signature):
    """Answer the question with a single word."""
    question: str = InputField()
    answer: str = OutputField()

def test_refine_forward_success_first_attempt():
    """Tests successful refinement on the first try."""
    lm = DummyLM([{"answer": "Brussels"}])
    dspy.settings.configure(lm=lm)
    reward_call_count = [0]
    def reward_fn(args, pred: Prediction) -> Prediction:
        reward_call_count[0] += 1
        answer_text = getattr(pred, 'answer', '') # Safely get the answer
        is_single_word = len(answer_text.split()) == 1
        score = 1.0 if is_single_word else 0.0
        feedback = "Good, single word." if is_single_word else "Answer should be a single word."
        return Prediction(score=score, feedback=feedback)
    refine = Refine(
        signature=SimpleQA,
        reward_fn=reward_fn,
        threshold=1.0,
        N=3,
        verbose=False
    )
    with patch.object(refine.predictor_module, 'forward', wraps=refine.predictor_module.forward) as mock_forward:
        result = refine(question="What is the capital of Belgium?")
        predictor_call_count = mock_forward.call_count
    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] == 1, "Reward function should have been called exactly once"
    assert predictor_call_count == 1, "Internal predictor should have been called exactly once"
    assert hasattr(result, "Refine_metadata"), "Result should have Refine_metadata attribute"
    metadata = result.Refine_metadata
    assert metadata["refine_successful"] is True, "Metadata should indicate success"
    assert metadata["iterations_made"] == 1, "Metadata should show 1 iteration was made"
    assert metadata["best_score"] == 1.0, "Metadata should record the best score achieved"
    assert len(metadata["attempts_log"]) == 1, "Metadata attempts log should have 1 entry"
    assert metadata["attempts_log"][0]["score"] == 1.0


def test_refine_module_fail_always_below_threshold():
    """Tests the behavior when the score never meets the threshold and fail_on_threshold=True."""
    lm = DummyLM([
        {"answer": "City of Brussels"},
        {"answer": "Capital Region"},
        {"answer": "Brussels City"}
    ])
    dspy.settings.configure(lm=lm)
    reward_call_count = [0]
    def reward_fn_always_fail(args, pred: Prediction) -> Prediction:
        reward_call_count[0] += 1
        return Prediction(score=0.0, feedback="Deliberately failing score")
    refine = Refine(
        signature=SimpleQA,
        reward_fn=reward_fn_always_fail,
        threshold=1.0,
        N=3,
        fail_if_below_threshold=True,
        verbose=False
    )
    with patch.object(refine.predictor_module, 'forward', wraps=refine.predictor_module.forward) as mock_forward:
        with pytest.raises(ValueError) as excinfo:
            refine(question="What is the capital of Belgium?")
        predictor_call_count = mock_forward.call_count
    assert "failed to meet the score threshold" in str(excinfo.value), "Error message should indicate threshold failure"
    assert reward_call_count[0] == 3, "Reward function should have been called 3 times (N=3)"
    assert predictor_call_count == 3, "Internal predictor should have been called 3 times (N=3)"


def test_refine_module_succeed_on_last_attempt():
    """Tests successful refinement on the final attempt (Nth attempt)."""
    lm = DummyLM([
        {"answer": "City of Brussels"},
        {"answer": "Capital Region"},
        {"answer": "Brussels"}
    ])
    dspy.settings.configure(lm=lm)
    reward_call_count = [0]
    def reward_fn_single_word(args, pred: Prediction) -> Prediction:
        reward_call_count[0] += 1
        answer_text = getattr(pred, 'answer', '')
        is_single_word = len(answer_text.split()) == 1
        score = 1.0 if is_single_word else 0.0
        feedback = "Good" if is_single_word else "Need single word"
        return Prediction(score=score, feedback=feedback)
    refine = Refine(
        signature=SimpleQA,
        reward_fn=reward_fn_single_word,
        threshold=1.0,
        N=3,
        fail_if_below_threshold=False,
        verbose=False
    )
    with patch.object(refine.predictor_module, 'forward', wraps=refine.predictor_module.forward) as mock_forward, \
            patch.object(lm, 'copy', return_value=lm) as mock_lm_copy:
        result = refine(question="What is the capital of Belgium?")
        print("Result:", result)
        predictor_call_count = mock_forward.call_count
    assert result.answer == "Brussels", "Result should be 'Brussels' from the 3rd attempt"
    assert reward_call_count[0] == 3, "Reward function should have been called 3 times"
    assert predictor_call_count == 3, "Internal predictor should have been called 3 times"
    assert hasattr(result, "Refine_metadata"), "Result should have Refine_metadata attribute"
    metadata = result.Refine_metadata
    assert metadata["refine_successful"] is True, "Metadata should indicate success as threshold was met on last try"
    assert metadata["iterations_made"] == 3, "Metadata should show 3 iterations were made"
    assert metadata["best_score"] == 1.0, "Metadata should record the best score (1.0) from the last attempt"
    assert len(metadata["attempts_log"]) == 3, "Metadata attempts log should have 3 entries"
    assert metadata["attempts_log"][0]["score"] == 0.0
    assert metadata["attempts_log"][1]["score"] == 0.0
    assert metadata["attempts_log"][2]["score"] == 1.0


def test_refine_returns_best_effort_when_threshold_not_met():
    """Tests that Refine returns the best attempt even if the threshold is never met (when fail_if_below_threshold=False)."""
    lm = DummyLM([
        {"answer": "City of Brussels is the place"},
        {"answer": "Capital Region"},
        {"answer": "Brussels City area"}
    ])
    dspy.settings.configure(lm=lm)
    reward_call_count = [0]
    def reward_fn_graded(args, pred: Prediction) -> Prediction:
        reward_call_count[0] += 1
        answer_text = getattr(pred, 'answer', '')
        num_words = len(answer_text.split())
        if num_words == 1:
            score = 1.0
            feedback = "Perfect (1 word)"
        elif num_words == 2:
            score = 0.5
            feedback = "Okay (2 words)"
        else:
            score = 0.0
            feedback = "Too long"
        return Prediction(score=score, feedback=feedback)
    refine = Refine(
        signature=SimpleQA,
        reward_fn=reward_fn_graded,
        threshold=1.0,
        N=3,
        fail_if_below_threshold=False,
        verbose=False
    )
    with patch.object(refine.predictor_module, 'forward', wraps=refine.predictor_module.forward) as mock_forward, \
            patch.object(lm, 'copy', return_value=lm) as mock_lm_copy:
        result = refine(question="What is the capital of Belgium?")
        predictor_call_count = mock_forward.call_count
    assert result.answer == "Capital Region", "Result should be the one with the highest score (0.5)"
    assert reward_call_count[0] == 3, "Reward function should have been called 3 times"
    assert predictor_call_count == 3, "Internal predictor should have been called 3 times"
    assert hasattr(result, "Refine_metadata"), "Result should have Refine_metadata attribute"
    metadata = result.Refine_metadata
    assert metadata["refine_successful"] is False, "Metadata should indicate failure to meet threshold"
    assert metadata["iterations_made"] == 3, "Metadata should show 3 iterations were made"
    assert metadata["best_score"] == 0.5, "Metadata should record the best score achieved (0.5)"
    assert len(metadata["attempts_log"]) == 3, "Metadata attempts log should have 3 entries"
    assert metadata["attempts_log"][0]["score"] == 0.0
    assert metadata["attempts_log"][1]["score"] == 0.5
    assert metadata["attempts_log"][2]["score"] == 0.0