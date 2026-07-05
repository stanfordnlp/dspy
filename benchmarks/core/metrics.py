"""
Metric definitions and registry for different datasets.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import TYPE_CHECKING, Any, Callable, Dict

import dspy
from dspy.evaluate.metrics import hotpot_f1_score

if TYPE_CHECKING:
    from dspy import Example, Prediction


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and removing articles/punctuation."""
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_yesno(text: str) -> str | None:
    """Extract yes/no from a verbose answer if it starts with yes/no."""
    normalized = normalize_text(text)
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return None


def smart_hotpot_f1_score(
    prediction: str,
    ground_truth: str,
    should_extract_yesno: bool = False,
) -> float:
    """Smart HotPotQA F1 score that handles verbose yes/no answers.

    For yes/no questions, extracts the yes/no prefix from verbose predictions.
    For other questions, uses standard token-level F1 scoring.

    Args:
        prediction: Predicted answer.
        ground_truth: Reference answer.
        should_extract_yesno: Whether to extract yes/no from verbose predictions.

    Returns:
        F1 score in [0.0, 1.0].
    """
    norm_pred = normalize_text(prediction)
    norm_gold = normalize_text(ground_truth)

    # Handle yes/no/noanswer special cases
    yesno_answers = {"yes", "no", "noanswer"}
    if norm_gold in yesno_answers and should_extract_yesno:
        extracted = extract_yesno(norm_pred)
        if extracted is not None:
            return 1.0 if extracted == norm_gold else 0.0

    return hotpot_f1_score(norm_pred, norm_gold)


def _get_answer(obj: Any, attr: str = "answer") -> str:
    """Safely extract answer attribute from an object."""
    return getattr(obj, attr, "") or ""


def _compute_f1_score(pred_answer: str, gold_answer: str | list[str]) -> float:
    """Compute F1 score, handling both single and multi-reference answers.

    `should_extract_yesno=True` is required for HotPotQA yes/no questions:
    `hotpot_f1_score` returns 0 when the gold is `"yes"` / `"no"` / `"noanswer"`
    and the prediction is not the bare word, even when the prediction is
    semantically correct (e.g. `"No, both individuals are not film producers."`
    vs gold `"no"`).
    """
    if isinstance(gold_answer, list):
        return max(
            smart_hotpot_f1_score(pred_answer, ans, should_extract_yesno=True)
            for ans in gold_answer
        )
    return smart_hotpot_f1_score(pred_answer, gold_answer, should_extract_yesno=True)


# Standard metric functions
def hotpotqa_metric(example: Example, pred: Prediction, trace: Any = None) -> float:
    """Standard HotPotQA metric for evaluation.

    Uses smart F1 scoring that handles verbose yes/no answers.
    Compatible with dspy.Evaluate and most optimizers.
    """
    pred_answer = _get_answer(pred)
    gold_answer = _get_answer(example)
    return _compute_f1_score(pred_answer, gold_answer)


def hotpotqa_metric_gepa(
    gold: Example,
    pred: Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> Prediction:
    """GEPA-compatible metric with feedback for reflective optimization.

    GEPA requires (gold, pred, trace, pred_name, pred_trace) signature and
    returns a Prediction with score and feedback fields.
    """
    pred_answer = _get_answer(pred)
    gold_answer = _get_answer(gold)
    score = _compute_f1_score(pred_answer, gold_answer)

    # Provide feedback for GEPA's reflection
    if score < 1.0:
        feedback = f"Expected: '{gold_answer}', Got: '{pred_answer}'. Score: {score:.2f}"
    else:
        feedback = "Perfect match!"

    return dspy.Prediction(score=score, feedback=feedback)


def exact_match_metric(example: Example, pred: Prediction, trace: Any = None) -> float:
    """Exact match metric for evaluation."""
    pred_answer = normalize_text(_get_answer(pred))
    gold_answer = normalize_text(_get_answer(example))
    return 1.0 if pred_answer == gold_answer else 0.0


def exact_match_metric_gepa(
    gold: Example,
    pred: Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> Prediction:
    """GEPA-compatible exact match metric."""
    pred_answer = _get_answer(pred)
    gold_answer = _get_answer(gold)

    pred_normalized = normalize_text(pred_answer)
    gold_normalized = normalize_text(gold_answer)
    score = 1.0 if pred_normalized == gold_normalized else 0.0

    feedback = "Perfect match!" if score == 1.0 else f"Expected: '{gold_answer}', Got: '{pred_answer}'"
    return dspy.Prediction(score=score, feedback=feedback)


def aime_metric(example: Example, pred: Prediction, trace: Any = None) -> float:
    """Standard AIME metric for evaluation.

    Exact match on integer answers. Parses both prediction and gold answer as integers.
    """
    correct_answer = _get_answer(example)
    pred_answer = _get_answer(pred)

    try:
        correct_int = int(correct_answer)
    except ValueError:
        # If gold answer isn't an integer, fall back to string comparison
        return 1.0 if pred_answer == correct_answer else 0.0

    try:
        pred_int = int(pred_answer)
    except ValueError:
        # If prediction isn't parseable as integer, it's wrong
        return 0.0

    return 1.0 if correct_int == pred_int else 0.0


def aime_metric_gepa(
    gold: Example,
    pred: Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> Prediction:
    """GEPA-compatible metric with feedback for AIME math problems.

    Provides detailed feedback including the solution when available.
    """
    correct_answer = _get_answer(gold)
    pred_answer = _get_answer(pred)
    written_solution = getattr(gold, 'solution', '')

    try:
        correct_int = int(correct_answer)
    except ValueError:
        # If gold answer isn't an integer, provide error feedback
        feedback = f"The correct answer '{correct_answer}' couldn't be parsed as an integer."
        return dspy.Prediction(score=0.0, feedback=feedback)

    try:
        pred_int = int(pred_answer)
    except ValueError:
        # Prediction couldn't be parsed as integer
        feedback_text = (
            f"The final answer must be a valid integer and nothing else. "
            f"You responded with '{pred_answer}', which couldn't be parsed as a python integer. "
            f"Please ensure your answer is a valid integer without any additional text or formatting."
        )
        feedback_text += f" The correct answer is '{correct_answer}'."

        if written_solution:
            feedback_text += (
                f" Here's the full step-by-step solution:\n{written_solution}\n\n"
                f"Think about what takeaways you can learn from this solution to improve your "
                f"future answers and approach to similar problems and ensure your final answer "
                f"is a valid integer."
            )

        return dspy.Prediction(score=0.0, feedback=feedback_text)

    score = 1.0 if correct_int == pred_int else 0.0

    # Provide appropriate feedback
    if score == 1.0:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += (
            f" Here's the full step-by-step solution:\n{written_solution}\n\n"
            f"Think about what takeaways you can learn from this solution to improve your "
            f"future answers and approach to similar problems."
        )

    return dspy.Prediction(score=score, feedback=feedback_text)


class MetricRegistry:
    """Registry for dataset-specific metrics."""

    _metrics: Dict[str, Dict[str, Callable]] = {
        "hotpotqa": {
            "standard": hotpotqa_metric,
            "gepa": hotpotqa_metric_gepa
        },
        "exact_match": {
            "standard": exact_match_metric,
            "gepa": exact_match_metric_gepa
        },
        "aime": {
            "standard": aime_metric,
            "gepa": aime_metric_gepa
        }
    }
    
    @classmethod
    def get_metric(cls, dataset_name: str, metric_type: str = "standard") -> Callable:
        """Get metric function for dataset.
        
        Args:
            dataset_name: Name of the dataset.
            metric_type: Type of metric ("standard" or "gepa").
            
        Returns:
            Metric function.
            
        Raises:
            ValueError: If dataset or metric type not found.
        """
        if dataset_name not in cls._metrics:
            available = ", ".join(cls._metrics.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        dataset_metrics = cls._metrics[dataset_name]
        if metric_type not in dataset_metrics:
            available = ", ".join(dataset_metrics.keys())
            raise ValueError(f"Unknown metric type: {metric_type}. Available: {available}")
        
        return dataset_metrics[metric_type]
    
    @classmethod
    def register_metric(cls, dataset_name: str, metric_type: str, metric_func: Callable) -> None:
        """Register a new metric.
        
        Args:
            dataset_name: Name of the dataset.
            metric_type: Type of metric ("standard" or "gepa").
            metric_func: Metric function.
        """
        if dataset_name not in cls._metrics:
            cls._metrics[dataset_name] = {}
        
        cls._metrics[dataset_name][metric_type] = metric_func
    
    @classmethod
    def list_datasets(cls) -> list[str]:
        """List available datasets."""
        return list(cls._metrics.keys())
    
    @classmethod
    def list_metric_types(cls, dataset_name: str) -> list[str]:
        """List available metric types for a dataset."""
        if dataset_name not in cls._metrics:
            return []
        return list(cls._metrics[dataset_name].keys())