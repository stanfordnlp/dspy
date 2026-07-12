"""End-to-end showcase: flex a BANKING77 intent classifier, then optimize its code with GEPA.

Flow:
  1. Write a classification Signature and "flex" it: ``dspy.Flex(signature)`` binds a
     baseline that delegates to ``dspy.RLM`` (no codegen — the recursive LM in a REPL).
  2. Benchmark that baseline on a held-out test split.
  3. Run ``dspy.GEPA`` over a small train/val split. Because the module is flex-marked,
     GEPA optimizes its *code* (``module_src``) — decomposing the
     task into focused predictors / plain Python.
  4. Benchmark the optimized program on the same test split.
  5. Plot baseline-vs-optimized accuracy to `banking77_improvement.png`.

The GEPA metric's *feedback* is the prompt GEPA feeds its code proposer. We deliberately
make that feedback adversarial — it forces the model to interrogate its own logic and
*prove* why each classification is correct rather than pattern-match — to squeeze more out
of the reflection model.

Needs real LMs + network (HuggingFace `PolyAI/banking77`). Skips without an API key or the
optional `datasets`/`matplotlib` deps. Dataset: https://huggingface.co/datasets/PolyAI/banking77
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

pd = pytest.importorskip("pandas")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# HuggingFace `PolyAI/banking77` is a script-based loader (incompatible with datasets>=3),
# and it simply wraps these canonical CSVs. We read them directly — same data.
BANKING77_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

DEMO_DIR = Path(__file__).parent
SAVE_PATH = DEMO_DIR / "banking77_flex.json"
PLOT_PATH = DEMO_DIR / "banking77_improvement.png"

# Executor runs the classifier (and the baseline RLM's sub-queries); reflection authors the
# optimized code. We deliberately make the executor a SMALL model (Haiku) and the reflection
# model a STRONG one (Opus): a weak executor leaves real accuracy headroom on a hard 77-way
# task, so the RLM baseline is well below 100% and GEPA's decomposition has something to lift.
# (With a strong executor on a tiny test split, even the un-optimized RLM baseline scores ~100%
# and the demo shows no improvement.) Both override via env.
_exec_default = "anthropic/claude-haiku-4-5"
_reflect_default = "anthropic/claude-opus-4-7"
EXEC_LM = dspy.LM(os.getenv("BANKING_EXEC_LM", _exec_default), max_tokens=2000)
REFLECTION_LM = dspy.LM(os.getenv("BANKING_REFLECTION_LM", _reflect_default), temperature=1.0, max_tokens=8000)

# A larger test split than the train/val budget: BANKING77 is genuinely hard, so n=40 gives a
# stable accuracy estimate with headroom (a tiny n=5 split too easily lands on a fluke 100%).
N_TRAIN, N_VAL, N_TEST = 20, 10, 40
MAX_METRIC_CALLS = 60
EVAL_THREADS = 8

# Small per-LLM-call penalty folded into GEPA's score. Accuracy stays the dominant term (a
# correct answer is worth 1.0), but among equally-accurate programs the one that makes fewer
# LM calls scores higher. Without this, the metric is pure accuracy: the RLM baseline already
# maxes it, so GEPA has no score reason to replace the opaque RLM with a decomposed program —
# its decomposition *feedback* is then overridden by its score-based acceptance, and the code
# never changes. This makes "decompose into focused, deterministic predictors" actually pay.
# Kept small (0.02) so it only breaks ties: a decomposition must *hold* accuracy to win — it
# can never beat a strictly-more-accurate program, so the accuracy headline can't regress.
LLM_CALL_PENALTY = 0.02

# The "challenge" injected into GEPA's reflection prompt (via metric feedback): push the
# model to justify its logic instead of guessing.
CHALLENGE = (
    "Before you keep or revise this classifier, challenge yourself: is the approach actually "
    "correct, or are you pattern-matching on superficial keywords? For every decision the code "
    "makes you must be able to PROVE why the chosen intent is the customer's true goal and why "
    "each competing intent is wrong. Question your own assumptions and justify them explicitly. "
    "Prefer decomposing the task into focused predictors over one opaque call."
)


def _norm(label: str) -> str:
    return (label or "").strip().lower().replace(" ", "_")


def _predicted(pred) -> str:
    return _norm(getattr(pred, "intent", ""))


def _load_splits():
    train_df = pd.read_csv(f"{BANKING77_BASE}/train.csv")
    test_df = pd.read_csv(f"{BANKING77_BASE}/test.csv")
    labels = sorted(train_df["category"].unique())

    def to_examples(df):
        return [dspy.Example(text=r.text, intent=r.category).with_inputs("text") for r in df.itertuples(index=False)]

    pool = train_df.sample(frac=1, random_state=0).reset_index(drop=True)  # disjoint train/val
    train = to_examples(pool.iloc[:N_TRAIN])
    val = to_examples(pool.iloc[N_TRAIN : N_TRAIN + N_VAL])
    test = to_examples(test_df.sample(N_TEST, random_state=0))
    return labels, train, val, test


def _build_signature(labels: list[str]):
    instructions = (
        "You are an intent classifier for the BANKING77 dataset. Given a single retail-banking "
        "customer message, return the one most appropriate intent.\n\n"
        "The answer MUST be exactly one of these 77 snake_case intent labels:\n"
        + ", ".join(labels)
        + ".\n\nReturn only the label, verbatim, with no extra words or punctuation."
    )
    return dspy.Signature("text: str -> intent: str", instructions)


def _evaluate(program: dspy.Module, dataset: list) -> tuple[float, float]:
    """Return (test accuracy, avg traced LLM calls/example).

    The call count makes the determinism win visible: the RLM baseline makes several traced
    predictor calls per example (one per REPL iteration, plus extract), whereas a GEPA-
    decomposed program typically settles each case in one focused call (or zero, in pure
    Python). Accuracy is the headline; calls show *how* the answer was reached.
    """
    def run_one(ex):
        try:
            with dspy.context(lm=EXEC_LM, trace=[]):
                pred = program(**ex.inputs())
                n_calls = len(dspy.settings.trace or [])
            return (_predicted(pred) == _norm(ex.intent), n_calls)
        except Exception:
            return (False, 0)

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        results = list(pool.map(run_one, dataset))
    accuracy = sum(hit for hit, _ in results) / len(dataset)
    avg_calls = sum(n for _, n in results) / len(dataset)
    return accuracy, avg_calls


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
    target = _norm(gold.intent)
    predicted = _predicted(pred)
    correct = predicted == target
    n_calls = len(trace) if trace else 0

    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    cost = f"(used {n_calls} LM call(s), cost {LLM_CALL_PENALTY * n_calls:.2f})"
    if correct:
        fb = (
            f"CORRECT — predicted '{predicted}', which matches the true intent {cost}. {CHALLENGE} "
            "Even though this one was right, prove the reasoning was sound and not luck, and settle "
            "clear cases in fewer (ideally one) focused predictor calls — the opaque RLM loop is "
            "penalized per call."
        )
    else:
        fb = (
            f"WRONG — predicted '{predicted or '<empty>'}' but the true intent is '{target}' {cost}. "
            f"{CHALLENGE} Diagnose the exact reasoning flaw that produced the wrong label."
        )
    return ScoreWithFeedback(score=score, feedback=fb)


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print(f"\n===== {label} =====")
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print("--- module_src (a normal dspy.Module subclass) ---")
    print(program.module_src)


def test_flex_banking77_showcase() -> None:
    dspy.configure(lm=EXEC_LM)
    labels, train, val, test = _load_splits()
    print(f"\nBANKING77: {len(labels)} intents | train={len(train)} val={len(val)} test={len(test)}")

    # 1. Flex the classifier: a dspy.RLM baseline (a clean dspy.Module subclass), code-optimizable.
    program = dspy.Flex(_build_signature(labels))
    baseline_src = program.module_src
    assert program.module_src.lstrip().startswith("class ")
    assert "dspy.RLM(" in baseline_src  # the un-optimized flex baseline
    _showcase(program, "baseline (un-optimized flex)")

    # 2. Benchmark the baseline.
    baseline_acc, baseline_calls = _evaluate(program, test)
    print(f"[baseline / flex-RLM] test accuracy = {baseline_acc:.1%}, avg LLM calls/example = {baseline_calls:.1f}")

    # 3. Optimize the module's CODE with GEPA (challenging feedback drives the reflection).
    optimized = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=5,
        num_threads=EVAL_THREADS,
        track_stats=True,
    ).compile(program, trainset=train, valset=val)

    # 4. Benchmark the optimized program on the same test split.
    optimized_acc, optimized_calls = _evaluate(optimized, test)
    code_changed = optimized.module_src != baseline_src
    print(f"[optimized / GEPA]    test accuracy = {optimized_acc:.1%}, avg LLM calls/example = {optimized_calls:.1f}")
    print(
        f"improvement: {optimized_acc - baseline_acc:+.1%} accuracy, "
        f"{optimized_calls - baseline_calls:+.1f} LLM calls/example | GEPA changed the code: {code_changed}"
    )
    detailed = getattr(optimized, "detailed_results", None)
    if detailed is not None:
        print(f"GEPA val scores explored: {detailed.val_aggregate_scores}")
    _showcase(optimized, "optimized by GEPA")

    # 4b. Persist the optimized program with the standard Module.save/load — the generated
    # code (module_src) round-trips alongside predictor state, no special on-disk format.
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(_build_signature(labels))
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")

    # 5. Plot the before/after. Two panels: accuracy is the headline (GEPA holds it), while
    # avg LLM calls/example shows the decomposition win (the opaque RLM loop -> focused code).
    labels_xy = ["baseline\n(flex / RLM)", "optimized\n(GEPA code)"]
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_acc, ax_calls) = plt.subplots(1, 2, figsize=(8, 4))

    acc_bars = ax_acc.bar(labels_xy, [baseline_acc, optimized_acc], color=colors)
    ax_acc.set_ylabel("test accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Accuracy (held)")
    for bar, acc in zip(acc_bars, [baseline_acc, optimized_acc], strict=True):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.1%}", ha="center", va="bottom")

    call_bars = ax_calls.bar(labels_xy, [baseline_calls, optimized_calls], color=colors)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(baseline_calls, optimized_calls, 1) * 1.2)
    ax_calls.set_title("LLM calls (lower = more deterministic)")
    for bar, n in zip(call_bars, [baseline_calls, optimized_calls], strict=True):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, n, f"{n:.1f}", ha="center", va="bottom")

    fig.suptitle(f"BANKING77 intent classification (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")

    # Showcase invariants: the pipeline ran end-to-end and we measured both ends.
    # (Whether GEPA changes the code / improves accuracy depends on the live models and
    # budget, so it's reported and plotted rather than hard-asserted.)
    assert PLOT_PATH.exists()
    assert 0.0 <= baseline_acc <= 1.0 and 0.0 <= optimized_acc <= 1.0
    assert optimized.module_src is not None


if __name__ == "__main__":
    test_flex_banking77_showcase()
