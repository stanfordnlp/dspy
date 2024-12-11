# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import dsp
from dsp import normalize_text
from dspy.evaluate import SemanticF1


def answer_exact_match(example, pred, trace=None, frac=1.0):
    assert(type(example.answer) is str or type(example.answer) is list)
    
    if type(example.answer) is str:
        return dsp.answer_match(pred.answer, [example.answer], frac=frac)
    else: # type(example.answer) is list
        return dsp.answer_match(pred.answer, example.answer, frac=frac)

answer_exact_match_str = dsp.answer_match

def answer_passage_match(example, pred, trace=None):
    assert(type(example.answer) is str or type(example.answer) is list)
    
    if type(example.answer) is str:
        return dsp.passage_match(pred.context, [example.answer])
    else: # type(example.answer) is list
        return dsp.passage_match(pred.context, example.answer)

def answer_similar_match(example, pred, trace=None):
    assert(type(example.answer) is str or type(example.answer) is list)
    
    def is_substring(text1, text2):
        # Normalize both texts using the existing normalize_text function
        text1 = normalize_text(text1)
        text2 = normalize_text(text2)
        return text1 in text2 or text2 in text1
    
    pred_answer = pred.answer
    if type(example.answer) is str:
        return is_substring(pred_answer, example.answer)
    else:  # type(example.answer) is list
        return any(is_substring(pred_answer, ans) for ans in example.answer)

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

