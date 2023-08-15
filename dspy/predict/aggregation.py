from dspy.primitives.prediction import Prediction, Completions
from dsp.utils import normalize_text


default_normalize = lambda s: normalize_text(s) or None


def majority(prediction_or_completions, normalize=default_normalize, field=None):
    """
        Returns the most common completion for the target field (or the last field) in the signature.
        When normalize returns None, that completion is ignored.
    """

    assert isinstance(prediction_or_completions, Prediction) or isinstance(prediction_or_completions, Completions)
    input_type = type(prediction_or_completions)

    # Get the completions
    if isinstance(prediction_or_completions, Prediction):
        completions = prediction_or_completions.completions
    else:
        completions = prediction_or_completions
    
    # Normalize
    field = field if field else completions.signature.fields[-1].output_variable
    normalize = normalize if normalize else lambda x: x
    normalized_values = [normalize(completion[field]) for completion in completions]
    normalized_values_ = [x for x in normalized_values if x is not None]
    
    # Count
    value_counts = {}
    for value in (normalized_values_ or normalized_values):
        value_counts[value] = value_counts.get(value, 0) + 1

    majority_value = max(value_counts, key=value_counts.get)

    # Return the first completion with the majority value in the field
    for completion in completions:
        if normalize(completion[field]) == majority_value:
            break
    
    if input_type == Prediction:
        return Prediction.from_completions([completion], signature=completions.signature)

    return Completions([completion], signature=completions.signature)

