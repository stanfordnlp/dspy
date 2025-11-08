from typing import Any, Callable, Union
from dspy.predict.aggregation import majority
from dspy.primitives.prediction import Prediction, Completions

def majority_k(
    predict_fn: Union[Callable[..., Any], Prediction, Completions, list], 
    k: int = 5, 
    **majority_kwargs
) -> Any:
    """
    Minimal wrapper running predict_fn k times and returning majority().
    
    Args:
        predict_fn: A callable that takes keyword arguments and returns a value,
                   or an existing Prediction/Completions/list to pass to majority().
        k: Number of times to run the predictor (only used if predict_fn is callable).
        **majority_kwargs: Additional arguments to pass to majority().
        
    Returns:
        If predict_fn is callable: A callable that runs predict_fn k times and returns the majority result.
        Otherwise: The result of majority(predict_fn, **majority_kwargs).
    """
    if not callable(predict_fn):
        return majority(predict_fn, **majority_kwargs)

    def wrapped(**inputs: Any) -> Any:
        preds = [predict_fn(**inputs) for _ in range(k)]
        return majority(preds, **majority_kwargs)

    return wrapped
