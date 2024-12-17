from dspy.dsp.utils import EM, normalize_text

from dspy.evaluate import auto_evaluation
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate import metrics

__all__ = [
    "auto_evaluation",
    "Evaluate",
    "metrics",
    "EM",
    "normalize_text",
]
