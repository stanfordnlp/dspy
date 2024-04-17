import re
import string
import unicodedata
from collections import Counter

import dspy
from dspy.evaluate.dpr import DPR_normalize
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction


def normalize_text(s):
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(prediction, answers_list):
    return max(em_score(prediction, ans) for ans in answers_list)


def F1(prediction, answers_list):
    return max(f1_score(prediction, ans) for ans in answers_list)


def em_score(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        # Unlike most tasks, QReCC and SQuAD-2.0 assign 1.0 in this edge case. We don't for uniformity.
        dspy.logger.info("F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.")

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def answer_match(prediction, answers, frac=1.0):
    if frac >= 1.0:
        EM(prediction, answers)

    return F1(prediction, answers) >= frac


def passage_match(passages, answers):
    return any(passage_has_answers(psg, answers) for psg in passages)


def passage_has_answers(passage: str, answers: list[str]) -> bool:
    return has_answer(
        tokenized_answers=[DPR_normalize(normalize_text(ans)) for ans in answers],
        text=normalize_text(passage),
    )


def answer_exact_match(example: Example, pred: Prediction, trace=None, frac: float = 1.0):
    assert type(example.answer) is str or type(example.answer) is list

    if type(example.answer) is str:
        return answer_match(pred.answer, [example.answer], frac=frac)
    else:  # type(example.answer) is list
        return answer_match(pred.answer, example.answer, frac=frac)


answer_exact_match_str = answer_match


def answer_passage_match(example: Example, pred: Prediction, trace=None):
    assert type(example.answer) is str or type(example.answer) is list

    if type(example.answer) is str:
        return passage_match(pred.context, [example.answer])
    else:  # type(example.answer) is list
        return passage_match(pred.context, example.answer)
