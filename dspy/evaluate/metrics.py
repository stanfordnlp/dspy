# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match


def _passage_match(passages: list[str], answers: list[str]) -> bool:
    """Returns True if any of the passages contains the answer."""

    from dspy.dsp.utils import passage_has_answers

    return any(passage_has_answers(psg, answers) for psg in passages)


def _answer_match(prediction, answers, frac=1.0):
    """Returns True if the prediction matches any of the answers."""

    from dspy.dsp.utils import EM, F1

    if frac >= 1.0:
        return EM(prediction, answers)

    return F1(prediction, answers) >= frac


def answer_exact_match(example, pred, trace=None, frac=1.0):
    if isinstance(example.answer, str):
        return _answer_match(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        return _answer_match(pred.answer, example.answer, frac=frac)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")


def answer_passage_match(example, pred, trace=None):
    if isinstance(example.answer, str):
        return _passage_match(pred.context, [example.answer])
    elif isinstance(example.answer, list):
        return _passage_match(pred.context, example.answer)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")


def answer_exact_match_and_semantic(example, pred, trace=None, frac=1.0, threshold=0.95):
    """
    Combines exact match and semantic F1 score checks.
    Returns True if either exact match succeeds or semantic F1 score is above threshold.
    """
    # Check exact match first
    exact_match = answer_exact_match(example, pred, trace=trace, frac=frac)

    if exact_match:
        return True

    # If no exact match, check semantic similarity
    from dspy.evaluate import SemanticF1

    semantic_f1 = SemanticF1(threshold=threshold)
    semantic_score = semantic_f1(example, pred, trace=True)

    return semantic_score
