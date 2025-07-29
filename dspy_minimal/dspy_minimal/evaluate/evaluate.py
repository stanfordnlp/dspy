"""Evaluation functionality for DSPy minimal implementation."""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ..primitives.module import Module
from ..utils.exceptions import DSPyError

logger = logging.getLogger(__name__)


class Evaluate(Module):
    """Evaluation module for assessing model performance."""
    
    def __init__(self, metric: Union[str, Callable], **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.results = []
    
    def forward(self, **kwargs):
        """Evaluate the given inputs using the specified metric."""
        if isinstance(self.metric, str):
            return self._evaluate_with_builtin_metric(self.metric, **kwargs)
        elif callable(self.metric):
            return self._evaluate_with_custom_metric(self.metric, **kwargs)
        else:
            raise DSPyError(f"Metric must be a string or callable, got {type(self.metric)}")
    
    def _evaluate_with_builtin_metric(self, metric_name: str, **kwargs):
        """Evaluate using built-in metrics."""
        if metric_name == "exact_match":
            return self._exact_match(**kwargs)
        elif metric_name == "contains":
            return self._contains(**kwargs)
        elif metric_name == "f1":
            return self._f1_score(**kwargs)
        else:
            raise DSPyError(f"Unknown built-in metric: {metric_name}")
    
    def _evaluate_with_custom_metric(self, metric_func: Callable, **kwargs):
        """Evaluate using a custom metric function."""
        try:
            result = metric_func(**kwargs)
            self.results.append(result)
            return result
        except Exception as e:
            logger.error(f"Error in custom metric evaluation: {e}")
            raise DSPyError(f"Custom metric evaluation failed: {e}")
    
    def _exact_match(self, **kwargs):
        """Exact match evaluation."""
        prediction = kwargs.get("prediction", "")
        target = kwargs.get("target", "")
        
        if isinstance(prediction, (list, tuple)):
            prediction = " ".join(str(p) for p in prediction)
        if isinstance(target, (list, tuple)):
            target = " ".join(str(t) for t in target)
        
        result = str(prediction).strip().lower() == str(target).strip().lower()
        self.results.append(result)
        return result
    
    def _contains(self, **kwargs):
        """Contains evaluation."""
        prediction = kwargs.get("prediction", "")
        target = kwargs.get("target", "")
        
        if isinstance(prediction, (list, tuple)):
            prediction = " ".join(str(p) for p in prediction)
        if isinstance(target, (list, tuple)):
            target = " ".join(str(t) for t in target)
        
        result = str(target).lower() in str(prediction).lower()
        self.results.append(result)
        return result
    
    def _f1_score(self, **kwargs):
        """Simple F1 score evaluation."""
        prediction = kwargs.get("prediction", "")
        target = kwargs.get("target", "")
        
        if isinstance(prediction, (list, tuple)):
            prediction = " ".join(str(p) for p in prediction)
        if isinstance(target, (list, tuple)):
            target = " ".join(str(t) for t in target)
        
        pred_words = set(str(prediction).lower().split())
        target_words = set(str(target).lower().split())
        
        if not pred_words or not target_words:
            result = 0.0
        else:
            precision = len(pred_words & target_words) / len(pred_words)
            recall = len(pred_words & target_words) / len(target_words)
            
            if precision + recall == 0:
                result = 0.0
            else:
                result = 2 * (precision * recall) / (precision + recall)
        
        self.results.append(result)
        return result
    
    def get_results(self) -> List[Any]:
        """Get all evaluation results."""
        return self.results
    
    def get_average_score(self) -> float:
        """Get the average score from all results."""
        if not self.results:
            return 0.0
        
        numeric_results = [r for r in self.results if isinstance(r, (int, float))]
        if not numeric_results:
            return 0.0
        
        return sum(numeric_results) / len(numeric_results)
    
    def reset(self):
        """Reset evaluation results."""
        self.results = [] 