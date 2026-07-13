"""PAJAMA-style "program-as-a-judge" with dspy.Flex + dspy.GEPA on a pairwise-preference benchmark.

PAJAMA replaces LLM-as-a-judge with an LLM-synthesized Python judge: score each response in code
(structure, relevance, readability, ...) and pick the higher one, so judging runs locally, far
cheaper, and stays auditable. dspy.Flex fits because dspy.GEPA can rewrite a Flex module's code, not
just its prompts, so optimizing a Flex judge is literally program synthesis: it starts from an RLM
(LLM-as-a-judge) baseline and, guided by accuracy plus a per-LLM-call penalty, evolves toward
deterministic Python that calls the LLM only for genuinely ambiguous pairs.

This is the single-program version, not the paper's committee (which selects + calibrates ~8-21
programs per dataset and routes hard pairs to an LLM). A single pure-code judge is far weaker than the
committee: in our own ensemble runs, individual GEPA judges scored ~54-62% on JudgeLM. The paper
reports only the committee (81.13% on JudgeLM, Table 4) and does NOT publish individual-program
accuracies; its 81% comes from ensembling (abstention + selection + a label model — see
pajama_findings.md), not any single strong program. So judge a single program on cost, not on beating
the LLM: expect GEPA to codify a judge well below the RLM's accuracy but at orders-of-magnitude lower
cost. LLM_CALL_PENALTY / N_VAL / REFLECTION_MINIBATCH are the knobs between "hold accuracy (keep RLM)"
and "force codify".

Dataset is selectable via PAJAMA_DATASET (judgelm | prometheus); the gold winner comes from each set's
scores (ties dropped, classes balanced so chance is 50%). Needs real LMs and network (HuggingFace,
cached after the first run), and skips without an API key or the optional datasets/matplotlib deps.
Improvement depends on the live models and budget, so results are printed and plotted, not asserted.
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

datasets = pytest.importorskip("datasets")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

DEMO_DIR = Path(__file__).parent
# Which PAJAMA preference dataset to reproduce. JudgeLM keeps its original artifact names; any other
# dataset gets a suffix so its cache/outputs never clobber JudgeLM's. Streamers are in _STREAMERS below.
DATASET = os.getenv("PAJAMA_DATASET", "judgelm").lower()
_DATASET_LABELS = {"judgelm": "JudgeLM", "prometheus": "Prometheus"}
if DATASET not in _DATASET_LABELS:
    raise ValueError(f"PAJAMA_DATASET must be one of {sorted(_DATASET_LABELS)}, got {DATASET!r}")
DATASET_LABEL = _DATASET_LABELS[DATASET]
_suffix = "" if DATASET == "judgelm" else f"_{DATASET}"
DATA_PATH = DEMO_DIR / f"{DATASET}_pairs.jsonl"  # cached balanced sample (downloaded once)
SAVE_PATH = DEMO_DIR / f"pajama_flex{_suffix}.json"
PLOT_PATH = DEMO_DIR / f"pajama_improvement{_suffix}.png"

# The executor runs the judge (RLM sub-queries plus any LLM fallback the optimized code keeps); the
# reflection model writes the optimized judging code. A small executor leaves accuracy headroom for
# the baseline; a strong reflection model writes better logic. Override either via env.
_exec_default = "anthropic/claude-haiku-4-5"
_reflect_default = "anthropic/claude-opus-4-8"
EXEC_LM = dspy.LM(os.getenv("PAJAMA_EXEC_LM", _exec_default), max_tokens=8000)
REFLECTION_LM = dspy.LM(os.getenv("PAJAMA_REFLECTION_LM", _reflect_default), temperature=1.0, max_tokens=8000)

# Balanced splits (equal A/B wins) so chance is 50%. Deliberately larger than the other demos: a
# single code judge easily overfits a small val set (latching onto verbosity), so wide train/val let
# GEPA keep only a judge that generalizes. All env-overridable (floored to even to stay balanced).
# A full run at these defaults is ~$15-25 (dominated by the RLM evals; once GEPA codifies to pure
# Python the later evals are free); expect a single pure-code judge well below the RLM (the paper
# reports only the committee, not per-program accuracy).
N_TRAIN = int(os.getenv("PAJAMA_N_TRAIN", "200")) // 2 * 2
N_VAL = int(os.getenv("PAJAMA_N_VAL", "100")) // 2 * 2
N_TEST = int(os.getenv("PAJAMA_N_TEST", "120")) // 2 * 2
MAX_METRIC_CALLS = int(os.getenv("PAJAMA_BUDGET", "600"))  # must be several x N_VAL or GEPA never proposes
REFLECTION_MINIBATCH = int(os.getenv("PAJAMA_MINIBATCH", "8"))  # wide, same anti-overfit reason as val
EVAL_THREADS = 8
MAX_RESP_CHARS = 2000  # truncate each response to bound prompt size

# Per-LLM-call penalty folded into GEPA's score: the knob trading accuracy for cost. Accuracy still
# dominates per example, but the RLM judge's several calls per pair discount its score enough that a
# cheaper code judge holding most of the accuracy can outscore it. Lower it (<=0.02) to keep the LLM;
# raise it to codify harder.
LLM_CALL_PENALTY = 0.04

# PAJAMA's judging criteria (Sec. 2.2), passed to GEPA as code-proposer guidance.
CRITERIA = (
    "Encode the judging logic as Python that computes a quality score for EACH response and picks the "
    "higher one. Draw on PAJAMA's judging criteria, all computable in code: (1) Structure — headings, "
    "lists, transition markers, sentence/paragraph counts; (2) Relevance — lexical/semantic overlap "
    "with the question (e.g. token/keyword overlap, TF-IDF-style); (3) Readability — grammar, density, "
    "repetition; (4) Bias — see the WARNING below; (5) Factuality — internal consistency, presence of "
    "specifics, directly answering what was asked; (6) Safety — penalize refusals/harmful content. "
    "CRITICAL WARNING: do NOT let LENGTH or amount of formatting decide the winner — a naive 'longer / "
    "more markdown = better' judge scores BELOW CHANCE here, because the gold labels do not reward "
    "verbosity. Length-normalize your features and weight on-question relevance and correctness, not "
    "size. Reserve an LLM call ONLY for pairs the code genuinely cannot separate."
)


class PairwiseJudge(dspy.Signature):
    """Decide which of two candidate responses better answers the user's question.

    You are given a `question` and two responses, `response_a` and `response_b`. Return `winner` as
    exactly "A" or "B" — the response that is more helpful, relevant, correct, and appropriate.

    Prefer to encode the judging logic as a deterministic Python scoring function over the two
    responses and choose the higher-scoring one; reserve an LLM call only for genuinely ambiguous
    cases. Judge on substance, not length or formatting.
    """

    question: str = dspy.InputField(desc="The user's question or instruction.")
    response_a: str = dspy.InputField(desc="Candidate response A.")
    response_b: str = dspy.InputField(desc="Candidate response B.")
    winner: str = dspy.OutputField(desc='Which response is better: exactly "A" or "B".')


def _winner(pred) -> str:
    """Normalize a predicted winner to 'A' / 'B' (leniently), else ''.

    Handles clean labels ("A"/"B"), numeric ("1"/"2"), and phrasings like "Response A" / "Answer B".
    """
    w = str(getattr(pred, "winner", "") or "").strip().upper()
    if not w:
        return ""
    if w in ("A", "1"):
        return "A"
    if w in ("B", "2"):
        return "B"
    # Strip noise words that themselves contain an A/B (e.g. "ANSWER" has an A) before scanning.
    core = w
    for noise in ("RESPONSE", "ANSWER", "OPTION", "CHOICE", "BETTER", "IS", "THE"):
        core = core.replace(noise, "")
    has_a, has_b = "A" in core, "B" in core
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return w[0] if w[0] in ("A", "B") else ""


def _truncate(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= MAX_RESP_CHARS else text[:MAX_RESP_CHARS] + " …[truncated]"


def _stream_judgelm():
    """Yield balanced-choice pairs from JudgeLM-100K; winner from GPT-4 scores ``score=[s1, s2]``."""
    ds = datasets.load_dataset("BAAI/JudgeLM-100K", split="train", streaming=True)
    for row in ds:
        score = row.get("score")
        if not (isinstance(score, list) and len(score) == 2):
            continue
        s1, s2 = score
        if s1 == s2:
            continue  # drop ties -> clean binary choice
        q = (row.get("question_body") or "").strip()
        a = _truncate(row.get("answer1_body") or "")
        b = _truncate(row.get("answer2_body") or "")
        if q and a and b:
            yield {"question": q, "response_a": a, "response_b": b, "winner": "A" if s1 > s2 else "B"}


def _stream_prometheus():
    """Yield pairs from Prometheus Preference-Collection; winner from ``orig_score_A``/``orig_score_B``."""
    ds = datasets.load_dataset("prometheus-eval/Preference-Collection", split="train", streaming=True)
    for row in ds:
        try:
            sa, sb = int(row["orig_score_A"]), int(row["orig_score_B"])
        except (KeyError, TypeError, ValueError):
            continue
        if sa == sb:
            continue  # drop ties -> clean binary choice
        q = (row.get("orig_instruction") or "").strip()
        a = _truncate(row.get("orig_response_A") or "")
        b = _truncate(row.get("orig_response_B") or "")
        if q and a and b:
            yield {"question": q, "response_a": a, "response_b": b, "winner": "A" if sa > sb else "B"}


_STREAMERS = {"judgelm": _stream_judgelm, "prometheus": _stream_prometheus}


def _ensure_dataset(per_class: int) -> None:
    """Stream the selected dataset once and cache a balanced A-win/B-win sample as JSONL.

    Each streamer yields ``{question, response_a, response_b, winner}`` with the winner derived from
    the dataset's gold scores (ties dropped). We keep ``per_class`` of each so the cache is balanced.
    """
    if DATA_PATH.exists():
        rows = [line for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) >= 2 * per_class:
            return

    a_rows: list[dict] = []
    b_rows: list[dict] = []
    for pair in _STREAMERS[DATASET]():
        bucket = a_rows if pair["winner"] == "A" else b_rows
        if len(bucket) < per_class:
            bucket.append(pair)
        if len(a_rows) >= per_class and len(b_rows) >= per_class:
            break

    sample = a_rows + b_rows
    if len(sample) < 2 * per_class:
        raise RuntimeError(f"{DATASET_LABEL} yielded only {len(sample)} balanced rows; need {2 * per_class}.")
    with DATA_PATH.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")


def _to_example(row: dict) -> dspy.Example:
    return dspy.Example(
        question=row["question"],
        response_a=row["response_a"],
        response_b=row["response_b"],
        winner=row["winner"],
    ).with_inputs("question", "response_a", "response_b")


def _load_splits() -> tuple[list, list, list]:
    """Balanced train/val/test drawn deterministically from the cached preference sample."""
    need_per_class = (N_TRAIN + N_VAL + N_TEST) // 2
    _ensure_dataset(need_per_class)

    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    a = [r for r in rows if r["winner"] == "A"]
    b = [r for r in rows if r["winner"] == "B"]
    rng = random.Random(0)
    rng.shuffle(a)
    rng.shuffle(b)

    def take(seq, start, count):
        return [_to_example(r) for r in seq[start : start + count]]

    tr_pc, va_pc, te_pc = N_TRAIN // 2, N_VAL // 2, N_TEST // 2
    train = take(a, 0, tr_pc) + take(b, 0, tr_pc)
    val = take(a, tr_pc, va_pc) + take(b, tr_pc, va_pc)
    test = take(a, tr_pc + va_pc, te_pc) + take(b, tr_pc + va_pc, te_pc)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
    """Reward correct preference; penalize LLM calls to push judging into deterministic code."""
    target = gold.winner
    predicted = _winner(pred)
    if predicted not in ("A", "B"):
        return ScoreWithFeedback(
            score=0.0,
            feedback='`winner` was missing/unparseable. Return dspy.Prediction(winner="A" or "B").',
        )
    n_calls = len(trace) if trace else 0
    correct = predicted == target
    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    cost = f"(used {n_calls} LLM call(s), cost {LLM_CALL_PENALTY * n_calls:.2f})"

    if correct and n_calls == 0:
        fb = f"CORRECT with NO LLM call — ideal, this is the target. {CRITERIA} Keep settling clear pairs in Python."
    elif correct:
        fb = (
            f"CORRECT but {cost} — the LLM judge is penalized per call. Replace it: write a "
            f"`judge(question, response)` Python scorer, score BOTH responses, and pick the higher; "
            f"call the LLM only when the two scores are near-tied. {CRITERIA}"
        )
    else:
        fb = (
            f"WRONG — chose '{predicted}', gold is '{target}' {cost}. Diagnose which quality "
            f"difference your logic missed or over-weighted. {CRITERIA} "
            "If a pair is truly ambiguous for rules, route it to a single LLM judgment instead."
        )
    return ScoreWithFeedback(score=score, feedback=fb)


def _evaluate(program: dspy.Module, dataset: list) -> tuple[float, float]:
    """Return (accuracy, avg traced LLM calls per example) on the dataset."""

    def run_one(ex):
        try:
            with dspy.context(lm=EXEC_LM, trace=[]):
                pred = program(**ex.inputs())
                n_calls = len(dspy.settings.trace or [])
            return (_winner(pred) == ex.winner, n_calls)
        except Exception:
            return (False, 0)

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        results = list(pool.map(run_one, dataset))
    accuracy = sum(hit for hit, _ in results) / len(dataset)
    avg_calls = sum(n for _, n in results) / len(dataset)
    return accuracy, avg_calls


def _showcase(program: dspy.Module, label: str) -> None:
    print(f"\n===== {label} =====")
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print("--- module_src (a normal dspy.Module subclass) ---")
    print(program.module_src)


def test_flex_pajama_showcase() -> None:
    dspy.configure(lm=EXEC_LM)
    train, val, test = _load_splits()
    print(f"\nPAJAMA/{DATASET_LABEL} pairwise judge | train={len(train)} val={len(val)} test={len(test)} (balanced A/B)")

    # 1. Baseline: a Flex judge delegating to one dspy.RLM, i.e. LLM-as-a-judge.
    program = dspy.Flex(PairwiseJudge)
    baseline_src = program.module_src or ""
    assert baseline_src.lstrip().startswith("class ")
    assert "dspy.RLM(" in baseline_src
    _showcase(program, "baseline (LLM-as-a-judge / flex RLM)")

    base_acc, base_calls = _evaluate(program, test)
    print(f"[baseline / LLM-as-judge] accuracy = {base_acc:.1%}, avg LLM calls/example = {base_calls:.1f}")

    # 2. GEPA rewrites the judge's code toward a deterministic (or hybrid) judge.
    optimized = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=REFLECTION_MINIBATCH,
        num_threads=EVAL_THREADS,
        track_stats=True,
    ).compile(program, trainset=train, valset=val)

    opt_acc, opt_calls = _evaluate(optimized, test)
    code_changed = optimized.module_src != baseline_src
    print(f"[optimized / PAJAMA-style] accuracy = {opt_acc:.1%}, avg LLM calls/example = {opt_calls:.1f}")
    print(
        f"change: {opt_acc - base_acc:+.1%} accuracy, {opt_calls - base_calls:+.1f} LLM calls/example "
        f"| GEPA changed the code: {code_changed}"
    )
    print(
        "paper reference — "
        + {
            "judgelm": "JudgeLM (revised, Table 4): a selected+calibrated 21-program committee reaches "
            "81.13% at 99.2% coverage; the paper does not report individual-program accuracy. Expect this "
            "single judge well below that, trading accuracy for ~1000x lower cost.",
            "prometheus": "Prometheus (revised): an 8-program committee reaches 88.78%; the paper does not "
            "report individual-program accuracy. Expect this single judge well below that, trading "
            "accuracy for ~1000x lower cost.",
        }[DATASET]
    )
    _showcase(optimized, "optimized by GEPA (program-as-a-judge)")

    # 3. Persist and reload; the code round-trips via Module.save/load.
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(PairwiseJudge)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized judge -> {SAVE_PATH}")

    # 4. Plot before/after: accuracy (should hold) and LLM calls (should drop).
    labels_xy = ["baseline\n(LLM-as-judge)", "optimized\n(program-as-judge)"]
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_acc, ax_calls) = plt.subplots(1, 2, figsize=(8, 4))

    acc_bars = ax_acc.bar(labels_xy, [base_acc, opt_acc], color=colors)
    ax_acc.set_ylabel("test accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.axhline(0.5, ls=":", c="#c00", lw=1)
    ax_acc.text(1.5, 0.51, "chance (balanced)", ha="right", va="bottom", fontsize=8, color="#c00")
    ax_acc.set_title("Preference accuracy")
    for bar, acc in zip(acc_bars, [base_acc, opt_acc], strict=True):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.1%}", ha="center", va="bottom")

    call_bars = ax_calls.bar(labels_xy, [base_calls, opt_calls], color=colors)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(base_calls, opt_calls, 1) * 1.2)
    ax_calls.set_title("LLM calls (lower = cheaper / more codified)")
    for bar, n in zip(call_bars, [base_calls, opt_calls], strict=True):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, n, f"{n:.1f}", ha="center", va="bottom")

    fig.suptitle(f"PAJAMA-style program-as-a-judge on {DATASET_LABEL} (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")

    # The pipeline ran end to end; improvement depends on the live models/budget, so it's reported
    # and plotted rather than asserted.
    assert PLOT_PATH.exists()
    assert 0.0 <= base_acc <= 1.0 and 0.0 <= opt_acc <= 1.0
    assert optimized.module_src is not None


if __name__ == "__main__":
    test_flex_pajama_showcase()
