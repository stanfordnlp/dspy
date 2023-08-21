# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import dsp
from dsp.utils import EM, normalize_text

def answer_exact_match(example, pred, trace=None, frac=1.0):
    return dsp.answer_match(pred.answer, [example.answer], frac=frac)

answer_exact_match_str = dsp.answer_match

def answer_passage_match(example, pred, trace=None):
    return dsp.passage_match(pred.context, [example.answer])

