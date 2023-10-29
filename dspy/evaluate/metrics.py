# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import dsp


def answer_exact_match(example, pred, trace=None, frac=1.0):
    assert isinstance(example.answer, str) or isinstance(example.answer, list)

    if isinstance(example.answer, str):
        return dsp.answer_match(pred.answer, [example.answer], frac=frac)
    else:  # type(example.answer) is list
        return dsp.answer_match(pred.answer, example.answer, frac=frac)


answer_exact_match_str = dsp.answer_match


def answer_passage_match(example, pred, trace=None):
    assert isinstance(example.answer, str) or isinstance(example.answer, list)

    if isinstance(example.answer, str):
        return dsp.passage_match(pred.context, [example.answer])
    else:  # type(example.answer) is list
        return dsp.passage_match(pred.context, example.answer)
