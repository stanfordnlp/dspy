from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

# dspy registers a lazy numpy proxy in sys.modules; matplotlib's `from numpy.exceptions import ...`
# trips that proxy into a recursive import. Materialize the real numpy first. (banking77/pajama get
# this for free by importing pandas/datasets before matplotlib; this demo depends on neither.)
np = pytest.importorskip("numpy")
_ = np.ndarray  # force the proxy to load the real module before matplotlib imports it
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

DEMO_DIR = Path(__file__).parent
DATA_PATH = DEMO_DIR / "conflation_coded.jsonl"
SAVE_PATH = DEMO_DIR / "conflation_flex.json"
PLOT_PATH = DEMO_DIR / "conflation_improvement.png"

EXEC_LM = dspy.LM("anthropic/claude-opus-4-7", max_tokens=1000)
STRONG_LM = dspy.LM("anthropic/claude-opus-4-7", max_tokens=8000)
dspy.configure(lm=EXEC_LM)

# A larger, class-balanced test split (40 = 20 pos + 20 neg) gives a stable accuracy/cost estimate
# with headroom; a tiny n=8 too easily lands on a fluke. Chance stays 50% (balanced). Splits are
# disjoint slices of the shuffled pools, so neg usage (8+5+20=33) stays well under the 260 available.
N_TRAIN_POS, N_TRAIN_NEG = 8, 8
N_VAL_POS, N_VAL_NEG = 5, 5
N_TEST_POS, N_TEST_NEG = 20, 20
MAX_METRIC_CALLS = 45
EVAL_THREADS = 8


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
    distance: float = dspy.InputField(desc="Distance between the two coordinates.")
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


def _evaluate(program: dspy.Module, dataset: list) -> tuple[float, float, float]:
    """Return (mean metric score, accuracy, avg LLM calls/example).

    The headline number is the *metric score* the optimizer actually optimizes — accuracy
    minus the 0.15-per-LLM-call penalty — not raw accuracy. The un-optimized RLM baseline
    classifies correctly but burns several traced LLM calls per example, so its score is well
    below 1.0 even at ~100% accuracy. GEPA's win is settling clear cases in deterministic
    Python (0 calls -> full score), which raw accuracy alone can't show.
    """
    def run_one(ex):
        try:
            with dspy.context(trace=[]):
                pred = program(**ex.inputs())
                trace = list(dspy.settings.trace or [])
            score = float(metric(ex, pred, trace=trace).score)
            ok = _as_bool(pred.is_same) == bool(ex.is_same)
            return score, int(ok), len(trace)
        except Exception:
            return 0.0, 0, 0

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        results = list(pool.map(run_one, dataset))
    n = len(dataset)
    total_score = sum(s for s, _, _ in results)
    correct = sum(ok for _, ok, _ in results)
    calls = sum(c for _, _, c in results)
    return total_score / n, correct / n, calls / n


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print(program.module_src)


def test_loader():
    program = dspy.Flex(SamePlace)
    program.load(str(SAVE_PATH))
    return program


def test_flex_conflation() -> None:
    dspy.configure(lm=EXEC_LM)
    train, val, test = _load_splits()
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")

    program = dspy.Flex(SamePlace)

    # Fresh baseline: a clean dspy.Module subclass that delegates to one dspy.RLM.
    assert program.module_src.lstrip().startswith("class ")
    assert "dspy.RLM(" in program.module_src
    _showcase(program, "baseline (un-optimized flex)")

    base_score, base_acc, base_calls = _evaluate(program, test)
    print(
        f"[baseline] score={base_score:.2f} "
        f"(accuracy={base_acc:.2f}, avg LLM calls/example={base_calls:.2f})"
    )

    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=STRONG_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=3,
        num_threads=EVAL_THREADS,
    ).compile(program, trainset=train, valset=val)

    # The metric penalizes LLM calls, so GEPA should push most logic into plain Python —
    # watch the module_src shift from one RLM to focused predictors + deterministic code.
    _showcase(optimized, "optimized by GEPA")
    print(f"GEPA changed the code: {optimized.module_src != program.module_src}")

    opt_score, opt_acc, opt_calls = _evaluate(optimized, test)
    print(
        f"[optimized] score={opt_score:.2f} "
        f"(accuracy={opt_acc:.2f}, avg LLM calls/example={opt_calls:.2f})"
    )
    print(f"score improvement: {opt_score - base_score:+.2f}")

    # Persist the optimized program with the standard Module.save/load (code round-trips).
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(SamePlace)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")

    # Plot the before/after (a la banking77), one panel per metric. Score is the headline GEPA
    # optimizes (accuracy − 0.15/LLM-call); accuracy shows it's held; LLM calls/example shows the
    # decomposition win (the opaque RLM loop -> deterministic Python that settles clear cases).
    labels_xy = ["baseline\n(flex / RLM)", "optimized\n(GEPA code)"]
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_score, ax_acc, ax_calls) = plt.subplots(1, 3, figsize=(11, 4))

    score_bars = ax_score.bar(labels_xy, [base_score, opt_score], color=colors)
    ax_score.set_ylabel("mean metric score")
    ax_score.set_ylim(0, 1.1)  # headroom so the on-bar labels clear the title at ~1.0
    ax_score.set_title("Score (accuracy − call penalty)")
    for bar, s in zip(score_bars, [base_score, opt_score], strict=True):
        ax_score.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center", va="bottom")

    acc_bars = ax_acc.bar(labels_xy, [base_acc, opt_acc], color=colors)
    ax_acc.set_ylabel("test accuracy")
    ax_acc.set_ylim(0, 1.1)  # headroom so the on-bar labels clear the title at 100%
    ax_acc.set_title("Accuracy (held)")
    for bar, a in zip(acc_bars, [base_acc, opt_acc], strict=True):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, a + 0.02, f"{a:.1%}", ha="center", va="bottom")

    call_bars = ax_calls.bar(labels_xy, [base_calls, opt_calls], color=colors)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(base_calls, opt_calls, 1) * 1.2)
    ax_calls.set_title("LLM calls (lower = more deterministic)")
    for bar, n in zip(call_bars, [base_calls, opt_calls], strict=True):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, n, f"{n:.1f}", ha="center", va="bottom")

    fig.suptitle(f"Conflation: same-place matching (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")
    assert PLOT_PATH.exists()


if __name__ == "__main__":
    test_flex_conflation()
