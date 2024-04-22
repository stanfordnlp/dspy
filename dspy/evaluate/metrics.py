from collections import Counter

import dspy
from dspy.evaluate.dpr import DPR_normalize, has_answer
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.utils import normalize_text


def em(prediction: str, answers_list: list[str]) -> bool:
    return max(em_score(prediction, ans) for ans in answers_list)


def f1(prediction: str, answers_list: list[str]) -> float:
    return max(f1_score(prediction, ans) for ans in answers_list)


def em_score(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
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
    return (2 * precision * recall) / (precision + recall)


def answer_match(prediction: str, answers: list[str], frac: float = 0.9) -> bool:
    if frac >= 1.0:
        return em(prediction, answers)

    return f1(prediction, answers) >= frac


def passage_match(passages: str, answers: list[str]) -> bool:
    return any(passage_has_answers(psg, answers) for psg in passages)


def passage_has_answers(passage: str, answers: list[str]) -> bool:
    return has_answer(
        tokenized_answers=[DPR_normalize(normalize_text(ans)) for ans in answers],
        text=normalize_text(passage),
    )


def answer_exact_match(example: Example, pred: Prediction, frac: float = 0.90, *_args, **_kwargs) -> bool:
    if not isinstance(example.answer, (str, list)):
        raise ValueError("example.answer must be str or list")

    if isinstance(example.answer, str):
        return answer_match(pred.answer, [example.answer], frac=frac)

    return answer_match(pred.answer, example.answer, frac=frac)


def answer_passage_match(example: Example, pred: Prediction, *_args, **_kwargs):
    if not isinstance(example.answer, (str, list)):
        raise ValueError("example.answer must be str or list")

    if isinstance(example.answer, str):
        return passage_match(pred.context, [example.answer])

    return passage_match(pred.context, example.answer)
