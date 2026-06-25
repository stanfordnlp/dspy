from __future__ import annotations

import json
import random
from pathlib import Path

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

DEMO_DIR = Path(__file__).parent
DATA_PATH = DEMO_DIR / "coded.jsonl"
FLEX_PATH = DEMO_DIR / "conflation_flex_gen.py"

EXEC_LM = dspy.LM("anthropic/claude-opus-4-7", max_tokens=1000)
STRONG_LM = dspy.LM("anthropic/claude-opus-4-7", max_tokens=8000)
dspy.configure(lm=EXEC_LM)

N_TRAIN_POS, N_TRAIN_NEG = 6, 6
N_VAL_POS, N_VAL_NEG = 3, 3
N_TEST_POS, N_TEST_NEG = 4, 4
MAX_METRIC_CALLS = 30


class SamePlace(dspy.Signature):
    """Decide whether two business listings refer to the same physical place.

    You compare place A (input_name / input_address) with place B (match_name /
    match_address), plus the geographic distance between them.

    Prefer a deterministic Python algorithm.
    Reserve an LLM call only for ambiguous ones that simple logic can't decide confidently.
    """

    input_name: str = dspy.InputField(desc="Name of place A.")
    input_address: str = dspy.InputField(desc="Street address of place A.")
    match_name: str = dspy.InputField(desc="Name of place B.")
    match_address: str = dspy.InputField(desc="Street address of place B.")
    distance: float = dspy.InputField(
        desc="Distance between the two coordinates."
    )
    is_same: bool = dspy.OutputField(desc="True if A and B are the same physical place, else False.")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value)


def _to_example(row: dict) -> dspy.Example:
    return dspy.Example(
        input_name=row["input_name"],
        input_address=row["input_address"],
        match_name=row["match_name"],
        match_address=row["match_address"],
        distance=float(row["distance"]),
        is_same=(row["judgment"] == "true"),
    ).with_inputs("input_name", "input_address", "match_name", "match_address", "distance")


def _load_splits() -> tuple[list, list, list]:
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    pos = [r for r in rows if r["judgment"] == "true"]
    neg = [r for r in rows if r["judgment"] == "false"]
    rng = random.Random(0)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def take(seq, start, count):
        return [_to_example(r) for r in seq[start : start + count]]

    train = take(pos, 0, N_TRAIN_POS) + take(neg, 0, N_TRAIN_NEG)
    val = take(pos, N_TRAIN_POS, N_VAL_POS) + take(neg, N_TRAIN_NEG, N_VAL_NEG)
    test = take(pos, N_TRAIN_POS + N_VAL_POS, N_TEST_POS) + take(neg, N_TRAIN_NEG + N_VAL_NEG, N_TEST_NEG)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
    """Reward correct + deterministic, penalize LLM calls."""
    example, prediction = gold, pred
    llm_call_penalty = 0.15
    n_calls = len(trace) if trace else 0  # predictor calls during this forward()
    try:
        pred = _as_bool(prediction.is_same)
    except Exception:
        return ScoreWithFeedback(
            score=0.0,
            feedback="`is_same` was missing or unreadable. Return dspy.Prediction(is_same=<bool>).",
        )
    gold = bool(example.is_same)
    correct = pred == gold
    score = max(0.0, (1.0 if correct else 0.0) - llm_call_penalty * n_calls)

    if not correct:
        fb = (
            f"WRONG: predicted is_same={pred}, expected {gold}. Use the input fields (name, address, distance)"
            f"to ideally decide deterministically in Python whether the two location are the same."
        )
        if n_calls == 0:
            fb += " If this case is truly ambiguous for rules, route it to the LLM judge instead."
    elif n_calls > 0:
        fb = (
            f"Correct, but used {n_calls} LLM call(s) (cost {llm_call_penalty * n_calls:.2f}). If the "
            "normalized name/address similarity and distance already make this clear, decide it in "
            "Python and skip the LLM. Reserve LLM calls for genuinely ambiguous cases only."
        )
    else:
        fb = "Correct with no LLM call. This is great! Keep settling clear cases deterministically."
    return ScoreWithFeedback(score=score, feedback=fb)


def _evaluate(program: dspy.Module, dataset: list) -> tuple[float, float]:
    correct = 0
    calls = 0
    for ex in dataset:
        try:
            with dspy.context(trace=[]):
                pred = program(**ex.inputs())
                n = len(dspy.settings.trace or [])
            ok = _as_bool(pred.is_same) == bool(ex.is_same)
        except Exception:
            ok, n = False, 0
        correct += int(ok)
        calls += n
    return correct / len(dataset), calls / len(dataset)


def test_flex_conflation() -> None:
    dspy.configure(lm=EXEC_LM)
    train, val, test = _load_splits()
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")

    program = dspy.Flex(SamePlace, persist_to=str(FLEX_PATH), codegen_lm=STRONG_LM)
    base_acc, base_calls = _evaluate(program, test)
    print(f"[baseline] test accuracy={base_acc:.2f}, avg LLM calls/example={base_calls:.2f}")

    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=STRONG_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=3,
        num_threads=4,
    ).compile(program, trainset=train, valset=val)

    opt_acc, opt_calls = _evaluate(optimized, test)
    print(f"[optimized] test accuracy={opt_acc:.2f}  avg LLM calls/example={opt_calls:.2f}")


if __name__ == "__main__":
    test_flex_conflation()
