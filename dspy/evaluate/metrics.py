# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import internals
from internals.utils import EM, normalize_text

def answer_exact_match(example, pred, trace=None, frac=1.0):
    return internals.answer_match(pred.answer, [example.answer], frac=frac)

answer_exact_match_str = internals.answer_match

def answer_passage_match(example, pred, trace=None):
    return internals.passage_match(pred.context, [example.answer])

