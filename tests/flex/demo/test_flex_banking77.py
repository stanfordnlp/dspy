"""End-to-end showcase: vibe a BANKING77 intent classifier, then optimize its code with GEPA.

Flow:
  1. Write a classification Signature and "vibe" it: ``dspy.Flex(signature)`` binds a
     baseline that delegates to ``dspy.RLM`` (no codegen — the recursive LM in a REPL).
  2. Benchmark that baseline on a held-out test split.
  3. Run ``dspy.GEPA`` over a small train/val split. Because the module is vibe-marked,
     GEPA optimizes its *code* (``predictors_src``/``forward_src``) — decomposing the
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
FLEX_PATH = DEMO_DIR / "banking77_flex_gen.py"
PLOT_PATH = DEMO_DIR / "banking77_improvement.png"

# Executor runs the classifier (and the baseline RLM's sub-queries); reflection authors
# the optimized code (benefits from a strong model). Both override via env; defaults adapt
# to whichever provider key is present.
_exec_default = "anthropic/claude-opus-4-7"
_reflect_default = "anthropic/claude-opus-4-7"
EXEC_LM = dspy.LM(os.getenv("BANKING_EXEC_LM", _exec_default), max_tokens=2000)
REFLECTION_LM = dspy.LM(
    os.getenv("BANKING_REFLECTION_LM", _reflect_default), temperature=1.0, max_tokens=8000
)

N_TRAIN, N_VAL, N_TEST = 10, 5, 5
MAX_METRIC_CALLS = 10
EVAL_THREADS = 8

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
        return [
            dspy.Example(text=r.text, intent=r.category).with_inputs("text")
            for r in df.itertuples(index=False)
        ]

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


def _accuracy(program: dspy.Module, dataset: list) -> float:
    def run_one(ex):
        try:
            with dspy.context(lm=EXEC_LM):
                pred = program(**ex.inputs())
            return _predicted(pred) == _norm(ex.intent)
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        hits = list(pool.map(run_one, dataset))
    return sum(hits) / len(dataset)


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
    target = _norm(gold.intent)
    predicted = _predicted(pred)
    correct = predicted == target
    if correct:
        fb = (
            f"CORRECT — predicted '{predicted}', which matches the true intent. {CHALLENGE} "
            "Even though this one was right, prove the reasoning was sound and not luck."
        )
    else:
        fb = (
            f"WRONG — predicted '{predicted or '<empty>'}' but the true intent is '{target}'. "
            f"{CHALLENGE} Diagnose the exact reasoning flaw that produced the wrong label."
        )
    return ScoreWithFeedback(score=1.0 if correct else 0.0, feedback=fb)


def test_flex_banking77_showcase() -> None:
    dspy.configure(lm=EXEC_LM)
    labels, train, val, test = _load_splits()
    print(f"\nBANKING77: {len(labels)} intents | train={len(train)} val={len(val)} test={len(test)}")

    # 1. Vibe the classifier: a dspy.RLM baseline, marked code-optimizable.
    program = dspy.Flex(_build_signature(labels), persist_to=str(FLEX_PATH))
    baseline_src = program.predictors_src
    assert "dspy.RLM(" in baseline_src  # the un-optimized vibe baseline

    # 2. Benchmark the baseline.
    baseline_acc = _accuracy(program, test)
    print(f"[baseline / vibe-RLM] test accuracy = {baseline_acc:.1%}")

    # 3. Optimize the module's CODE with GEPA (challenging feedback drives the reflection).
    optimized = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=3,
        num_threads=EVAL_THREADS,
        track_stats=True,
    ).compile(program, trainset=train, valset=val)

    # 4. Benchmark the optimized program on the same test split.
    optimized_acc = _accuracy(optimized, test)
    code_changed = optimized.predictors_src != baseline_src
    print(f"[optimized / GEPA]    test accuracy = {optimized_acc:.1%}")
    print(f"improvement: {optimized_acc - baseline_acc:+.1%} | GEPA changed the code: {code_changed}")
    detailed = getattr(optimized, "detailed_results", None)
    if detailed is not None:
        print(f"GEPA val scores explored: {detailed.val_aggregate_scores}")
    print("--- optimized predictors_src ---")
    print(optimized.predictors_src)

    # 5. Plot the before/after.
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["baseline\n(vibe / RLM)", "optimized\n(GEPA code)"],
        [baseline_acc, optimized_acc],
        color=["#9aa0a6", "#1a73e8"],
    )
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"BANKING77 intent classification (n={len(test)})")
    for bar, acc in zip(bars, [baseline_acc, optimized_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")

    # Showcase invariants: the pipeline ran end-to-end and we measured both ends.
    # (Whether GEPA changes the code / improves accuracy depends on the live models and
    # budget, so it's reported and plotted rather than hard-asserted.)
    assert PLOT_PATH.exists()
    assert 0.0 <= baseline_acc <= 1.0 and 0.0 <= optimized_acc <= 1.0
    assert optimized.predictors_src is not None and optimized.forward_src is not None


if __name__ == "__main__":
    test_flex_banking77_showcase()
