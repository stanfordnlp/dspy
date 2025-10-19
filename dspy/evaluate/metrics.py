# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import re
import string
import unicodedata
from collections import Counter

from dspy.dsp.utils.utils import print_message


def EM(prediction, answers_list):  # noqa: N802
    """Return True if any reference exactly matches the prediction (after normalization).

    Normalization includes: Unicode NFD, lowercasing, punctuation removal,
    English article removal ("a", "an", "the"), and whitespace collapse.

    Args:
        prediction (str): Predicted answer string.
        answers_list (list[str]): List of reference answers.

    Returns:
        bool: True if any reference exactly equals the prediction after normalization;
            otherwise False.

    Example:
        >>> EM("The Eiffel Tower", ["Eiffel Tower", "Louvre"])
        True
        >>> EM("paris", ["Paris"])
        True
    """
    assert isinstance(answers_list, list)
    return max(em_score(prediction, ans) for ans in answers_list)


def F1(prediction, answers_list):  # noqa: N802
    """Maximum token-level F1 score of prediction against a list of references.

    The F1 is computed on whitespace-tokenized, normalized strings (same
    normalization as :func:`EM`). The function returns the maximum F1 over all
    provided references.

    Args:
        prediction (str): Predicted answer string.
        answers_list (list[str]): List of reference answers.

    Returns:
        float: The highest F1 score in [0.0, 1.0].

    Example:
        >>> round(F1("Eiffel Tower is in Paris", ["Paris"]), 2)
        0.33
    """
    assert isinstance(answers_list, list)
    return max(f1_score(prediction, ans) for ans in answers_list)


def HotPotF1(prediction, answers_list):  # noqa: N802
    """Maximum token-level F1 score with HotpotQA label handling.

    Like :func:`F1`, but if either the normalized prediction or reference is one
    of the special labels {"yes", "no", "noanswer"} and they differ, the score
    is 0. Otherwise standard token F1 is used.

    Args:
        prediction (str): Predicted answer.
        answers_list (list[str]): List of reference answers.

    Returns:
        float: The highest HotpotQA-style F1 score in [0.0, 1.0].

    Example:
        >>> HotPotF1("yes", ["no"])
        0.0
    """
    assert isinstance(answers_list, list)
    return max(hotpot_f1_score(prediction, ans) for ans in answers_list)


def normalize_text(s):
    """Normalize text for string and token comparisons.

    Steps:
        1) Unicode NFD normalization,
        2) lowercasing,
        3) punctuation removal,
        4) English article removal ("a", "an", "the"),
        5) whitespace collapse.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string.

    Example:
        >>> normalize_text("The,  Eiffel  Tower!")
        'eiffel tower'
    """
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\\b(a|an|the)\\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(prediction, ground_truth):
    """Exact-match (boolean) after normalization.

    Args:
        prediction (str): Predicted answer.
        ground_truth (str): Reference answer.

    Returns:
        bool: True if normalized strings are identical; otherwise False.

    Example:
        >>> em_score("Paris", "paris")
        True
    """
    return normalize_text(prediction) == normalize_text(ground_truth)


def f1_score(prediction, ground_truth):
    """Token-level F1 between prediction and reference (after normalization).

    Strings are normalized (see :func:`normalize_text`) and split by whitespace;
    F1 is computed from token precision and recall. If there is no token overlap,
    returns 0. If both sides are empty, a diagnostic message is printed but the
    score is still 0 for uniformity.

    Args:
        prediction (str): Predicted answer.
        ground_truth (str): Reference answer.

    Returns:
        float: F1 score in [0.0, 1.0].

    Example:
        >>> round(f1_score("the Eiffel Tower", "Eiffel Tower"), 2)
        1.0
    """
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        print_message("\\n#> F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\\n")

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def hotpot_f1_score(prediction, ground_truth):
    """HotpotQA-style token F1 with special labels.

    If either normalized string is in {"yes", "no", "noanswer"} and they are not
    equal, the score is 0. Otherwise compute standard token F1 after normalization.

    Args:
        prediction (str): Predicted answer.
        ground_truth (str): Reference answer.

    Returns:
        float: HotpotQA-style F1 score in [0.0, 1.0].

    Example:
        >>> hotpot_f1_score("no", "yes")
        0.0
    """
    normalized_prediction = normalize_text(prediction)
    normalized_ground_truth = normalize_text(ground_truth)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def precision_score(prediction, ground_truth):
    """Token-level precision of prediction against reference (after normalization).

    Precision is (# overlapping tokens) / (# tokens in prediction).
    If there is no token overlap, returns 0. If both sides are empty, a
    diagnostic message is printed but precision remains 0 for uniformity.

    Args:
        prediction (str): Predicted answer.
        ground_truth (str): Reference answer.

    Returns:
        float: Precision in [0.0, 1.0].

    Example:
        >>> precision_score("eiffel tower in paris", "eiffel tower")
        0.67
    """
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        print_message("\\n#> Precision Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\\n")

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    return precision


def _passage_match(passages: list[str], answers: list[str]) -> bool:
    """Return True if any passage contains any answer (normalized & DPR-normalized)."""
    from dspy.dsp.utils import DPR_normalize, has_answer

    def passage_has_answers(passage: str, answers: list[str]) -> bool:
        """Returns True if the passage contains the answer."""
        return has_answer(
            tokenized_answers=[DPR_normalize(normalize_text(ans)) for ans in answers],
            text=normalize_text(passage),
        )

    return any(passage_has_answers(psg, answers) for psg in passages)


def _answer_match(prediction, answers, frac=1.0):
    """Return True if prediction matches any answer.

    When ``frac >= 1.0``, require exact-match (EM). Otherwise return whether the
    maximum token-level F1 across answers is at least ``frac``.
    """
    if frac >= 1.0:
        return EM(prediction, answers)

    return F1(prediction, answers) >= frac


def answer_exact_match(example, pred, trace=None, frac=1.0):
    """Example/Prediction evaluator for answer strings with EM/F1 thresholding.

    If ``example.answer`` is a string, compare ``pred.answer`` against it.
    If it's a list, compare against any of the references.
    When ``frac >= 1.0`` (default), use exact-match; otherwise require that the
    maximum F1 across references is at least ``frac``.

    Args:
        example: Object with attribute ``answer`` (str or list[str]).
        pred: Object with attribute ``answer`` (str).
        trace: Unused; reserved for compatibility.
        frac (float, optional): Threshold in [0.0, 1.0]. ``1.0`` means EM.

    Returns:
        bool: True if the match condition holds; otherwise False.

    Example:
        >>> from types import SimpleNamespace
        >>> ex = SimpleNamespace(answer=["Eiffel Tower", "Louvre"])
        >>> pr = SimpleNamespace(answer="The Eiffel Tower")
        >>> answer_exact_match(ex, pr, frac=1.0)  # EM
        True
        >>> answer_exact_match(ex, pr, frac=0.5)  # F1 threshold
        True
    """
    if isinstance(example.answer, str):
        return _answer_match(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        return _answer_match(pred.answer, example.answer, frac=frac)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")


def answer_passage_match(example, pred, trace=None):
    """Return True if any passage in ``pred.context`` contains the answer(s).

    Strings are normalized (and passages also use DPR normalization internally).

    Args:
        example: Object with attribute ``answer`` (str or list[str]).
        pred: Object with attribute ``context`` (list[str]) containing passages.
        trace: Unused; reserved for compatibility.

    Returns:
        bool: True if any passage contains any reference answer; otherwise False.

    Example:
        >>> from types import SimpleNamespace
        >>> ex = SimpleNamespace(answer="Eiffel Tower")
        >>> pr = SimpleNamespace(context=["The Eiffel Tower is in Paris.", "..."])
        >>> answer_passage_match(ex, pr)
        True
    """
    if isinstance(example.answer, str):
        return _passage_match(pred.context, [example.answer])
    elif isinstance(example.answer, list):
        return _passage_match(pred.context, example.answer)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")
