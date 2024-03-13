from dspy.primitives.prediction import Prediction, Completions
from dsp.utils import normalize_text


default_normalize = lambda s: normalize_text(s) or None


def majority(prediction, normalize=default_normalize, field=None):
    """
    Returns the most common completion for the target field (or the last field) in the signature.
    When normalize returns None, that completion is ignored.
    In case of a tie, earlier completion are prioritized.
    """

    assert isinstance(prediction, Prediction)
    return prediction.get_majority(field=field, normalize=normalize)
