import typing as t
from collections import Counter

from dsp.utils import normalize_text
from dspy.primitives.prediction import Prediction

default_normalize = lambda s: normalize_text(s) or None


def majority(
    prediction: Prediction,
    normalize: t.Callable[[str], t.Optional[str]] = default_normalize,
    field: t.Optional[str] = None,
) -> Prediction:
    """
    Returns the most common completion for the target field (or the last field) in the signature.
    When normalize returns None, that completion is ignored.
    In case of a tie, earlier completion are prioritized.
    """

    if normalize is None:
        normalize = lambda x: x

    if field is None:
        field = list(prediction.completions.signature.output_fields.keys())[-1]

    predictions = [normalize(completion[field]) for completion in prediction.completions]
    predictions = Counter([x for x in predictions if x is not None])

    majority_class = list(predictions.keys())[0]

    pred = Prediction.from_completions(prediction.completions)
    pred._completions.filter(field, majority_class)

    return pred
