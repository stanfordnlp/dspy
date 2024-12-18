# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import dsp
from dspy.evaluate import SemanticF1


def answer_exact_match(example, pred, trace=None, frac=1.0):
    assert isinstance(example.answer, (str, list))

    if isinstance(example.answer, str):
        return dsp.answer_match(pred.answer, [example.answer], frac=frac)
    else:  # isinstance(example.answer, list)
        return dsp.answer_match(pred.answer, example.answer, frac=frac)


answer_exact_match_str = dsp.answer_match


def answer_passage_match(example, pred, trace=None):
    assert isinstance(example.answer, (str, list))

    if isinstance(example.answer, str):
        return dsp.passage_match(pred.context, [example.answer])
    else:  # isinstance(example.answer, list)
        return dsp.passage_match(pred.context, example.answer)


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
    semantic_f1 = SemanticF1(threshold=threshold)
    semantic_score = semantic_f1(example, pred, trace=True)

    return semantic_score
