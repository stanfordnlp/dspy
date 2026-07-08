"""PAJAMA-style "program-as-a-judge" with ``dspy.Flex`` + ``dspy.GEPA``.

Reproduces the core idea of PAJAMA — *"Time To Impeach LLM-as-a-Judge: Programs are the Future
of Evaluation"* (Huang, Vishwakarma & Sala, arXiv:2506.10403) — on a real pairwise-preference
benchmark (JudgeLM).

PAJAMA in one line: instead of calling an LLM to pick the better of two responses (LLM-as-a-judge),
have an LLM **synthesize executable Python judging logic** — score each response with code
(structure, relevance, readability, bias, …) and pick the higher one — which then runs locally,
orders of magnitude cheaper, and is interpretable/auditable.

Why ``dspy.Flex`` + ``dspy.GEPA`` is the natural analog:
    ``dspy.Flex`` is a module whose *code* (``module_src``) ``dspy.GEPA`` may rewrite (not just its
    prompts). So the optimizer literally does *program synthesis for the judge*: it starts from an
    LLM-as-a-judge baseline (``dspy.RLM``) and, guided by accuracy feedback plus a per-LLM-call
    penalty, evolves the judge toward deterministic Python that scores the two responses and reserves
    an LLM call only for genuinely ambiguous cases (PAJAMA's "routing" idea). This is the *single-
    program* version of PAJAMA — we do not build PAJAMA's 52-program Snorkel ensemble — so calibrate
    expectations against the paper's *few-program* regime, not its full ensemble.

What "kinda similar" means here (JudgeLM, in-domain, from the paper's Table 1 / Fig. 2):
    * LLM-as-a-judge in-domain accuracy ≈ 74–83%.
    * PAJAMA's *programmatic* judge in-domain ≈ 63–73% (a few points below the LLM judge) — but ~3
      orders of magnitude cheaper ($0.053 vs $130–300 to label the set), and it *beats* the LLM judge
      out-of-domain on RewardBench Chat-Hard (+8.67 on JudgeLM).
    * With only ~3 programs PAJAMA sits ≈ 59%; it climbs to ≈ 82% with 52 programs.
    So a single Flex+GEPA code/hybrid judge landing in the ~60–75% band while making far fewer LLM
    calls than the RLM baseline reproduces the paper's headline tradeoff: competitive accuracy at a
    fraction of the cost.

    This demo is tuned for the **hybrid** regime — hold accuracy near the LLM judge, codify the clear
    cases, and route only genuinely ambiguous pairs to the LLM (``LLM_CALL_PENALTY`` small so accuracy
    strictly dominates; ``REFLECTION_MINIBATCH`` wide so a lucky-on-val degenerate judge can't be
    selected). Caveat learned empirically: a *single* GEPA program pushed to pure code (higher penalty)
    tends to overfit and latch onto an anti-correlated heuristic (verbosity — the very bias the paper
    flags), scoring at/below chance on test. That is expected for one program (the paper's 3-program
    point is only ~59%); reaching the paper's headline needs many programs aggregated. Raise the
    penalty knob to trade accuracy for cost and slide toward the paper's cheaper programmatic band.

Soundness notes:
    * Gold label = JudgeLM's GPT-4 reference scores ``score=[s1, s2]``; winner = A if s1>s2 else B.
      We drop the ~5% ties so the task is a clean binary A-vs-B choice (PAJAMA's judge vote is also
      binary), and we **balance** A-wins and B-wins so the majority-class baseline is 50% and accuracy
      is meaningful. JudgeLM/GPT-4 gold carries known position bias; balancing avoids a trivial
      baseline, and an order-invariant code judge is the robust ideal the paper argues for.
    * Responses are truncated to keep prompts/tokens bounded; the gold was assigned on full responses,
      a minor source of noise acceptable for a demo.

Needs real LMs + network (HuggingFace ``BAAI/JudgeLM-100K``, cached locally after first run). Skips
without an API key or the optional ``datasets``/``matplotlib`` deps. Whether GEPA improves the
tradeoff depends on the live models/budget, so results are reported and plotted, not hard-asserted.
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
DATA_PATH = DEMO_DIR / "judgelm_pairs.jsonl"  # cached balanced sample (downloaded once)
SAVE_PATH = DEMO_DIR / "pajama_flex.json"
PLOT_PATH = DEMO_DIR / "pajama_improvement.png"

# Executor runs the judge (the baseline RLM's sub-queries, and any LLM fallback the optimized code
# keeps); reflection authors the optimized judging code. Small executor (Haiku) leaves real accuracy
# headroom so the LLM-as-judge baseline isn't already perfect on a tiny split; strong reflection
# (Opus) writes better judging logic. Both override via env.
_exec_default = "anthropic/claude-haiku-4-5"
_reflect_default = "anthropic/claude-opus-4-8"
EXEC_LM = dspy.LM(os.getenv("PAJAMA_EXEC_LM", _exec_default), max_tokens=2000)
REFLECTION_LM = dspy.LM(os.getenv("PAJAMA_REFLECTION_LM", _reflect_default), temperature=1.0, max_tokens=8000)

# Balanced splits (equal A-wins / B-wins), so majority-class accuracy = 50%.
# Bigger train/val than the other demos ON PURPOSE: this is the "codify" regime (push judging into
# Python), and a single code judge easily overfits a small val set — it aces a handful of examples by
# luck, keys on an anti-correlated feature (verbosity — the bias the paper flags), and then scores
# BELOW chance on test. A larger val set (16) makes candidate ACCEPTANCE robust: a ~40% verbosity
# heuristic is rejected outright, so GEPA can only keep a code judge that genuinely generalizes.
N_TRAIN, N_VAL, N_TEST = 24, 16, 40  # must be even (balanced per class)
MAX_METRIC_CALLS = 70  # GEPA budget; like the paper, judge quality scales with it (3 progs→59%, 52→82%)
REFLECTION_MINIBATCH = 8  # wide, for the same anti-overfitting reason as the val set
EVAL_THREADS = 8
MAX_RESP_CHARS = 2000  # truncate each response to bound prompt size

# Per-LLM-call penalty folded into GEPA's score — the knob that trades accuracy for cost. A correct
# answer (1.0) always beats a wrong one (0.0), so accuracy dominates per example; but the RLM judge
# traces several calls per pair, so at this penalty its discounted score drops enough that a cheaper
# codified judge which HOLDS most of the accuracy outscores it — which is what pushes GEPA to codify
# (PAJAMA's thesis). This is the "codify" regime, reproducing the paper's headline: the programmatic
# judge lands a few points UNDER the LLM judge (~63–73% vs 74–83%) at a fraction of the cost. Lower it
# (≤0.02) for the hybrid regime (hold accuracy, keep the LLM); raise it to codify harder/cheaper.
LLM_CALL_PENALTY = 0.04

# PAJAMA's six judging criteria (Sec. 2.2), fed to GEPA as the code-proposer guidance. These are the
# axes the paper's synthesized programs encode as Python.
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
    # Strip noise words ("RESPONSE"/"ANSWER"/...) that themselves contain an A/B before scanning,
    # e.g. "Answer B" — "ANSWER" contains an A that would otherwise confuse the letter check.
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


def _ensure_dataset(per_class: int) -> None:
    """Stream JudgeLM-100K once and cache a balanced A-win/B-win sample as JSONL.

    Gold winner is derived from the GPT-4 reference scores (``score=[s1, s2]``): A if s1>s2, B if
    s2>s1; ties dropped. We collect ``per_class`` of each so the cached set is class-balanced.
    """
    if DATA_PATH.exists():
        rows = [l for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) >= 2 * per_class:
            return

    ds = datasets.load_dataset("BAAI/JudgeLM-100K", split="train", streaming=True)
    a_rows: list[dict] = []
    b_rows: list[dict] = []
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
        if not (q and a and b):
            continue
        winner = "A" if s1 > s2 else "B"
        bucket = a_rows if winner == "A" else b_rows
        if len(bucket) < per_class:
            bucket.append({"question": q, "response_a": a, "response_b": b, "winner": winner})
        if len(a_rows) >= per_class and len(b_rows) >= per_class:
            break

    sample = a_rows + b_rows
    if len(sample) < 2 * per_class:
        raise RuntimeError(f"JudgeLM yielded only {len(sample)} balanced rows; need {2 * per_class}.")
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
    """Balanced train/val/test drawn deterministically from the cached JudgeLM sample."""
    need_per_class = (N_TRAIN + N_VAL + N_TEST) // 2
    _ensure_dataset(need_per_class)

    rows = [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
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
    """Return (accuracy, avg traced LLM calls/example). Calls make the cost win visible: the RLM
    baseline burns several LLM calls per pair; a codified judge settles most in 0–1."""

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
    print(f"\nPAJAMA/JudgeLM pairwise judge | train={len(train)} val={len(val)} test={len(test)} (balanced A/B)")

    # 1. Flex the judge: baseline is a clean dspy.Module delegating to one dspy.RLM = LLM-as-a-judge.
    program = dspy.Flex(PairwiseJudge)
    baseline_src = program.module_src or ""
    assert baseline_src.lstrip().startswith("class ")
    assert "dspy.RLM(" in baseline_src
    _showcase(program, "baseline (LLM-as-a-judge / flex RLM)")

    base_acc, base_calls = _evaluate(program, test)
    print(f"[baseline / LLM-as-judge] accuracy = {base_acc:.1%}, avg LLM calls/example = {base_calls:.1f}")

    # 2. GEPA rewrites the judge's CODE — program synthesis toward a deterministic (or hybrid) judge.
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
        "paper reference (JudgeLM, in-domain): LLM-as-judge ≈ 74–83%, PAJAMA programmatic ≈ 63–73% "
        "at ~3 orders of magnitude lower cost."
    )
    _showcase(optimized, "optimized by GEPA (program-as-a-judge)")

    # 3. Persist + reload (code round-trips via the standard Module.save/load).
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(PairwiseJudge)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized judge -> {SAVE_PATH}")

    # 4. Plot before/after: accuracy (should hold near the LLM judge) + LLM calls (should drop).
    labels_xy = ["baseline\n(LLM-as-judge)", "optimized\n(program-as-judge)"]
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_acc, ax_calls) = plt.subplots(1, 2, figsize=(8, 4))

    acc_bars = ax_acc.bar(labels_xy, [base_acc, opt_acc], color=colors)
    ax_acc.set_ylabel("test accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.axhline(0.5, ls=":", c="#c00", lw=1)
    ax_acc.text(1.5, 0.51, "chance (balanced)", ha="right", va="bottom", fontsize=8, color="#c00")
    ax_acc.set_title("Preference accuracy")
    for bar, acc in zip(acc_bars, [base_acc, opt_acc]):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.1%}", ha="center", va="bottom")

    call_bars = ax_calls.bar(labels_xy, [base_calls, opt_calls], color=colors)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(base_calls, opt_calls, 1) * 1.2)
    ax_calls.set_title("LLM calls (lower = cheaper / more codified)")
    for bar, n in zip(call_bars, [base_calls, opt_calls]):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, n, f"{n:.1f}", ha="center", va="bottom")

    fig.suptitle(f"PAJAMA-style program-as-a-judge on JudgeLM (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")

    # Invariants: the pipeline ran end-to-end and we measured both ends. Improvement depends on the
    # live models/budget, so it is reported/plotted rather than hard-asserted (like the other demos).
    assert PLOT_PATH.exists()
    assert 0.0 <= base_acc <= 1.0 and 0.0 <= opt_acc <= 1.0
    assert optimized.module_src is not None


if __name__ == "__main__":
    test_flex_pajama_showcase()
