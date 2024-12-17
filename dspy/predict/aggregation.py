from dspy.dsp.utils import normalize_text
from dspy.primitives.prediction import Completions, Prediction

def default_normalize(s):
    return normalize_text(s) or None


def majority(prediction_or_completions, normalize=default_normalize, field=None):
    """
        Returns the most common completion for the target field (or the last field) in the signature.
        When normalize returns None, that completion is ignored.
        In case of a tie, earlier completion are prioritized.
    """

    assert any(isinstance(prediction_or_completions, t) for t in [Prediction, Completions, list])
    type(prediction_or_completions)

    # Get the completions
    if isinstance(prediction_or_completions, Prediction):
        completions = prediction_or_completions.completions
    else:
        completions = prediction_or_completions
    
    try:
        signature = completions.signature
    except Exception:
        signature = None
    
    if not field:
        if signature:
            field = list(signature.output_fields.keys())[-1]
        else:
            field = list(completions[0].keys())[-1]

    # Normalize
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
    
    # if input_type == Prediction:
    return Prediction.from_completions([completion], signature=signature)


