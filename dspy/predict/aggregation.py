from dspy.evaluate import normalize_text
from dspy.primitives.prediction import Completions, Prediction


def default_normalize(s):
    return normalize_text(s) or None


def _is_empty_completions(completions) -> bool:
    """True when there are no completion rows to vote on.

    Empty Completions({}) should be rejected, but Completions.__len__ currently
    calls next() on an empty values iterator and raises StopIteration. Completions
    with declared fields but zero rows, such as Completions({"answer": []}), must
    also be treated as empty. Inspect the stored dict instead of truth-testing
    the object.
    """
    stored = getattr(completions, "_completions", None)
    if isinstance(stored, dict):
        if not stored:
            return True
        first = next(iter(stored.values()), [])
        return len(first) == 0
    try:
        return len(completions) == 0
    except Exception:
        return not completions


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
        if completions is None:
            completions = [prediction_or_completions]
    else:
        completions = prediction_or_completions

    if _is_empty_completions(completions):
        raise ValueError("majority() requires at least one completion")

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
    for value in normalized_values_ or normalized_values:
        value_counts[value] = value_counts.get(value, 0) + 1

    majority_value = max(value_counts, key=value_counts.get)

    # Return the first completion with the majority value in the field
    for completion in completions:
        if normalize(completion[field]) == majority_value:
            break

    # if input_type == Prediction:
    return Prediction.from_completions([completion], signature=signature)
