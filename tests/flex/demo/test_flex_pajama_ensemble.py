"""Multi-program aggregation for PAJAMA: GEPA synthesizes K diverse code judges, then their votes
are combined. This is the paper's actual method (52 programs + a Snorkel label model); here it's a
small first cut to see whether the accuracy-vs-#programs curve trends up.

The single-program demo (test_flex_pajama.py) shows why this is needed: one GEPA-synthesized judge
can't reach the paper's cheap-judge band, because a single program sits below the LLM judge by the
paper's own scaling (Fig. 2: 3 programs ~59%, 52 ~82%). The fix is aggregation — many cheap programs
that err in *different* ways, combined by vote so their independent mistakes cancel. Diversity here
comes from giving each judge a different criterion emphasis and a different GEPA seed; each is pushed
toward pure Python (no LLM calls), so the whole ensemble runs at near-zero cost.

Votes are combined by plain arithmetic (majority vote), but the ensemble weights come from a
dspy.Signature — an LLM "label model" that reads each judge's validation accuracy, coverage, and
mutual agreement and assigns a weight per judge: down-weight unreliable or redundant judges, flip a
below-chance one. That aggregator is itself GEPA-optimized on cross-val folds of the validation set,
so the whole pipeline — judges and label model — is tuned end to end. A closed-form 2*acc-1 rule is
the deterministic fallback if the aggregator's output can't be used.

Reuses the single-program demo's data, signature, gold labels, winner parsing, and baseline eval.
Needs real LMs + network and skips without an API key or the optional datasets/matplotlib deps. Cost
scales with N_JUDGES GEPA runs, so lower N_JUDGES via env for a faster run. Results are reported and
plotted, not asserted.
"""

from __future__ import annotations

import json
import math
import os
import random
import time

# Reuse the stable pieces of the single-program demo. Importing it also runs its importorskip guards
# (so this file skips too when datasets/matplotlib are missing) and hands back its already-configured
# Agg matplotlib as `plt`, so we never import pyplot before the backend is set.
from test_flex_pajama import (
    DATA_PATH,
    DEMO_DIR,
    EXEC_LM,
    REFLECTION_LM,
    PairwiseJudge,
    _ensure_dataset,
    _evaluate,
    _to_example,
    _winner,
    plt,
)

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

PLOT_PATH = DEMO_DIR / "pajama_ensemble.png"
SRC_PATH = DEMO_DIR / "pajama_ensemble.json"

_start = time.monotonic()


def _log(msg: str) -> None:
    """Timestamped, flushed progress line (visible live under `pytest -s`)."""
    print(f"[+{time.monotonic() - _start:5.0f}s] {msg}", flush=True)

# The paper SAMPLES ~52 programs (no per-program optimization) + a Snorkel label model. GEPA instead
# OPTIMIZES each program, so the thesis is FEWER-but-better judges: a handful of GEPA-optimized,
# decorrelated judges beating many sampled ones at matched #programs (i.e. our accuracy-vs-#programs
# curve sitting ABOVE the paper's). So keep N_JUDGES small and invest in per-judge independence and
# quality (bagging + enough train), not in judge count. Cost ~ baseline RLM eval + N_JUDGES * (RLM seed
# eval on N_VAL); dial N_JUDGES / N_VAL down via env to cut it.
N_JUDGES = int(os.getenv("PAJAMA_N_JUDGES", "6"))  # FEW GEPA-optimized judges (the point), not many sampled ones
N_TRAIN = int(os.getenv("PAJAMA_N_TRAIN", "32"))  # even; bigger train feeds bagging + curbs overfitting
N_VAL = int(os.getenv("PAJAMA_N_VAL", "20"))  # even; more val -> steadier weights, but drives per-judge cost
N_TEST = int(os.getenv("PAJAMA_N_TEST", "50"))  # even; less headline noise
BAG_FRAC = float(os.getenv("PAJAMA_BAG_FRAC", "0.7"))  # each judge trains on a different random slice of train
PER_JUDGE_BUDGET = 60  # GEPA budget per judge; must exceed one val-set eval (N_VAL) with room for
# several reflection rounds — too low and GEPA spends it all scoring the seed and never proposes code.
REFLECTION_MINIBATCH = 4
EVAL_THREADS = 8
JUDGE_PENALTY = 0.20  # per-LLM-call penalty; keeps each judge pure Python (like the paper's LLM-free scripts).
# A judge that routes one call to a genuinely ambiguous pair only loses ~0.20, so selective use survives.

# The aggregator (EnsembleWeighting) is itself GEPA-optimized: its instructions are tuned to turn each
# judge's validation stats into weights that generalize. Trained on cross-val folds of the val set so it
# learns a policy rather than one memorized weight vector. Toggle off to use the un-tuned signature.
OPTIMIZE_AGGREGATOR = os.getenv("PAJAMA_OPTIMIZE_AGGREGATOR", "1") == "1"
AGG_FOLDS = 4  # cross-val folds over the val set -> that many training examples for the aggregator
AGG_BUDGET = 12  # GEPA budget for the aggregator (cheap: its metric is pure arithmetic, no LLM calls)

# Each judge gets a different criterion emphasis AND a different bagged trainset (see _synthesize_judges) —
# aggregation only helps if judges err independently. The paper's diversity is 52 independently synthesized
# programs; here it's emphasis + a distinct data slice + a distinct GEPA seed. Bagging is the main lever
# (judges optimized on the SAME data converge on the same features, however different their emphasis).
JUDGE_EMPHASES = (
    "Weight RELEVANCE most: does the response actually answer the exact question asked?",
    "Weight FACTUALITY and SPECIFICITY most: concrete, correct, well-supported claims over vague ones.",
    "Weight STRUCTURE and READABILITY most: clarity, organization, grammar — strictly length-normalized.",
    "Weight DIRECTNESS most: reward answering head-on; penalize hedging, preamble, and non-answers.",
    "Weight SAFETY and REFUSAL-AVOIDANCE most: penalize unsafe content and unwarranted refusals.",
    "Weight COMPLETENESS most: cover every part of the question with concrete detail, without padding.",
    "Weight SIGNAL DENSITY most: information per token; penalize repetition, filler, and boilerplate.",
    "Weight COHERENCE most: logical flow and internal consistency; penalize contradictions and rambling.",
)

_BASE_INSTRUCTION = (
    'Decide which of two candidate responses better answers the user\'s question; return winner as exactly '
    '"A" or "B". Encode the judging entirely as a deterministic Python scoring function over the two '
    "responses and pick the higher-scoring one. Make NO LLM calls — this judge must be pure code."
)

# Judging guidance for the code proposer. Derived from the single-program demo's CRITERIA but pure-Python
# only (no LLM-fallback clause), since each ensemble judge must be a cheap program.
_GUIDANCE = (
    "Score EACH response in Python and pick the higher one; make no LLM calls. Use computable criteria: "
    "structure, question relevance (token/keyword overlap), readability, directness, and specificity. "
    "CRITICAL: do NOT let length or amount of markdown decide the winner — a 'longer = better' judge scores "
    "below chance here. Length-normalize your features and weight on-question relevance and correctness, "
    "not size. Do NOT hardcode phrases copied from specific examples or add special-cases for one task type "
    "(e.g. an acronym handler) — those overfit the small trainset and fail on held-out data. Prefer general, "
    "reusable features."
)


def _make_judge_metric(emphasis: str):
    """A GEPA metric that rewards a correct A/B vote and penalizes LLM calls, carrying the judge's
    emphasis and the pure-Python guidance as feedback."""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
        predicted = _winner(pred)
        if predicted not in ("A", "B"):
            return ScoreWithFeedback(score=0.0, feedback='Return dspy.Prediction(winner="A" or "B").')
        n_calls = len(trace) if trace else 0
        correct = predicted == gold.winner
        score = max(0.0, (1.0 if correct else 0.0) - JUDGE_PENALTY * n_calls)
        verdict = "CORRECT" if correct else f"WRONG (gold is '{gold.winner}')"
        fb = (
            f"{verdict}. This judge must be PURE Python with no LLM calls (this run used {n_calls}). "
            f"{emphasis} {_GUIDANCE}"
        )
        return ScoreWithFeedback(score=score, feedback=fb)

    return metric


def _bag(train: list, seed: int) -> list:
    """A different random slice of the trainset per judge (bagging). Judges optimized on the SAME data
    converge on the same features and err together; different slices decorrelate them — the whole point
    of the ensemble — and curb overfitting to any single example. Uses the full set if it's tiny."""
    k = max(4, int(len(train) * BAG_FRAC))
    if k >= len(train):
        return list(train)
    return random.Random(1000 + seed).sample(train, k)


def _synthesize_judges(train: list, val: list) -> list:
    """GEPA-synthesize N_JUDGES pure-Python judges — each with its own emphasis, GEPA seed, and a
    distinct bagged slice of the trainset so the judges err independently."""
    judges = []
    for i in range(N_JUDGES):
        emphasis = JUDGE_EMPHASES[i % len(JUDGE_EMPHASES)]
        bag = _bag(train, i)
        _log(f"synthesizing judge {i + 1}/{N_JUDGES} with GEPA ({emphasis.split(':')[0]}; {len(bag)} train)...")
        seed = PairwiseJudge.with_instructions(f"{_BASE_INSTRUCTION}\n\n{emphasis}")
        optimized = dspy.GEPA(
            metric=_make_judge_metric(emphasis),
            reflection_lm=REFLECTION_LM,
            max_metric_calls=PER_JUDGE_BUDGET,
            reflection_minibatch_size=REFLECTION_MINIBATCH,
            num_threads=EVAL_THREADS,
            seed=i,
        ).compile(dspy.Flex(seed), trainset=bag, valset=val)
        judges.append(optimized)
        codified = "dspy.RLM(" not in (optimized.module_src or "")
        _log(f"  judge {i + 1}/{N_JUDGES} done (codified to pure Python: {codified})")
    return judges


def _gold_vec(dataset: list) -> list[int]:
    """Gold winner per example as +1 (A) / -1 (B)."""
    return [1 if ex.winner == "A" else -1 for ex in dataset]


def _judge_votes(judge: dspy.Module, dataset: list) -> list[int]:
    """One judge's vote per example: +1 (A), -1 (B), 0 if unparseable."""
    votes = []
    for ex in dataset:
        try:
            with dspy.context(lm=EXEC_LM, trace=[]):
                w = _winner(judge(**ex.inputs()))
        except Exception:
            w = ""
        votes.append(1 if w == "A" else -1 if w == "B" else 0)
    return votes


class EnsembleWeighting(dspy.Signature):
    """Assign a weight to each judge in a weighted-vote ensemble from how it did on a validation set.

    Trust reliable judges, give a coin-flip judge about 0, and give a consistently-wrong judge a
    negative weight (which flips its vote). Also down-weight redundant judges: two that almost always
    agree add little over one. Return one weight per judge, in the same order as the report.
    """

    judge_report: str = dspy.InputField(
        desc="One line per judge: index, criterion emphasis, validation accuracy, coverage, and how "
        "often it agrees with each other judge."
    )
    weights: list[float] = dspy.OutputField(desc="One weight per judge in [-1, 1], same order as the report.")


def _weighting_report(val_votes: list[list[int]], val_gold: list[int], emphases: list[str]) -> str:
    """Per-judge validation stats (accuracy, coverage, pairwise agreement) fed to the weighting signature."""
    lines = []
    for i, votes in enumerate(val_votes):
        scored = [(v, g) for v, g in zip(votes, val_gold, strict=True) if v != 0]
        acc = sum(v == g for v, g in scored) / len(scored) if scored else 0.5
        coverage = sum(v != 0 for v in votes) / len(votes)
        agreements = []
        for j, other in enumerate(val_votes):
            if j == i:
                continue
            both = [(x, y) for x, y in zip(votes, other, strict=True) if x != 0 and y != 0]
            agree = sum(x == y for x, y in both) / len(both) if both else 0.0
            agreements.append(f"J{j}={agree:.0%}")
        emphasis = emphases[i].split(":")[0]
        lines.append(f"J{i} [{emphasis}] accuracy={acc:.0%} coverage={coverage:.0%} agrees({', '.join(agreements)})")
    return "\n".join(lines)


def _aggregator_trainset(val_votes: list[list[int]], val_gold: list[int], emphases: list[str], n_folds: int) -> list:
    """Cross-val examples for optimizing the aggregator: fit the judge report on the other folds, then
    score the weights it produces on the held-out fold. This gives GEPA a generalization signal rather
    than one report to memorize."""
    n = len(val_gold)
    examples = []
    for f in range(n_folds):
        hold = set(range(f, n, n_folds))  # round-robin fold
        fit = [i for i in range(n) if i not in hold]
        if not fit or not hold:
            continue
        report = _weighting_report([[v[i] for i in fit] for v in val_votes], [val_gold[i] for i in fit], emphases)
        examples.append(
            dspy.Example(
                judge_report=report,
                score_votes=[[v[i] for i in sorted(hold)] for v in val_votes],
                score_gold=[val_gold[i] for i in sorted(hold)],
            ).with_inputs("judge_report")
        )
    return examples


def _aggregator_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
    """Score a proposed weight vector by the ensemble accuracy it yields on the held-out fold."""
    n_judges = len(gold.score_votes)
    try:
        weights = [float(w) for w in pred.weights]
    except Exception:
        return ScoreWithFeedback(score=0.0, feedback=f"Return `weights` as a list of {n_judges} numbers in [-1, 1].")
    if len(weights) != n_judges or not all(math.isfinite(w) for w in weights):
        return ScoreWithFeedback(score=0.0, feedback=f"Return exactly {n_judges} finite weights, one per judge.")
    acc = _accuracy(_aggregate(gold.score_votes, weights), gold.score_gold)
    return ScoreWithFeedback(
        score=acc,
        feedback=(
            f"These weights scored {acc:.0%} ensemble accuracy on held-out validation examples. Raise the "
            "weight of judges that are accurate and add independent signal; lower or negate judges that are "
            "unreliable or redundant with others."
        ),
    )


def _optimize_aggregator(val_votes: list[list[int]], val_gold: list[int], emphases: list[str]):
    """GEPA-tune the aggregator's instructions on cross-val folds, or return the un-tuned signature when
    optimization is off or there aren't enough folds to learn from."""
    aggregator = dspy.ChainOfThought(EnsembleWeighting)
    trainset = _aggregator_trainset(val_votes, val_gold, emphases, AGG_FOLDS)
    if not OPTIMIZE_AGGREGATOR or len(trainset) < 2:
        return aggregator
    return dspy.GEPA(
        metric=_aggregator_metric,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=AGG_BUDGET,
        reflection_minibatch_size=min(AGG_FOLDS, len(trainset)),
        num_threads=EVAL_THREADS,
        seed=0,
    ).compile(aggregator, trainset=trainset, valset=trainset)


def _learned_weights(aggregator, val_votes: list[list[int]], val_gold: list[int], emphases: list[str]) -> list[float]:
    """Ensemble weights from the aggregator (the EnsembleWeighting label model, optionally GEPA-tuned)
    over the full validation report, with the closed-form 2*acc-1 rule as a deterministic fallback."""
    report = _weighting_report(val_votes, val_gold, emphases)
    try:
        with dspy.context(lm=EXEC_LM):
            out = aggregator(judge_report=report)
        weights = [float(w) for w in out.weights]
        if len(weights) == len(val_votes) and all(math.isfinite(w) for w in weights):
            return [max(-1.0, min(1.0, w)) for w in weights]
        print(f"  (aggregator gave {len(weights)} weights for {len(val_votes)} judges; using fallback)")
    except Exception as ex:
        print(f"  (aggregator failed: {ex}; using fallback)")
    return _reliability_weights(val_votes, val_gold)


def _reliability_weights(val_votes: list[list[int]], val_gold: list[int]) -> list[float]:
    """Deterministic fallback: 2 * (val accuracy) - 1 per judge (coin-flip -> 0, below-chance -> negative)."""
    weights = []
    for votes in val_votes:
        scored = [(v, g) for v, g in zip(votes, val_gold, strict=True) if v != 0]
        acc = sum(v == g for v, g in scored) / len(scored) if scored else 0.5
        weights.append(2 * acc - 1)
    return weights


def _aggregate(vote_matrix: list[list[int]], weights: list[float], k: int | None = None) -> list[int]:
    """Weighted vote over the first k judges: sign of the weighted sum, ties broken to A."""
    k = len(vote_matrix) if k is None else k
    n_examples = len(vote_matrix[0])
    return [1 if sum(weights[i] * vote_matrix[i][j] for i in range(k)) >= 0 else -1 for j in range(n_examples)]


def _accuracy(preds: list[int], gold: list[int]) -> float:
    return sum(p == g for p, g in zip(preds, gold, strict=True)) / len(gold)


def _splits() -> tuple[list, list, list]:
    """Balanced train/val/test drawn deterministically from the cached JudgeLM sample."""
    per_class = (N_TRAIN + N_VAL + N_TEST) // 2
    _ensure_dataset(per_class)
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    a = [r for r in rows if r["winner"] == "A"]
    b = [r for r in rows if r["winner"] == "B"]
    rng = random.Random(0)
    rng.shuffle(a)
    rng.shuffle(b)

    def take(seq, start, count):
        return [_to_example(r) for r in seq[start : start + count]]

    tr, va, te = N_TRAIN // 2, N_VAL // 2, N_TEST // 2
    train = take(a, 0, tr) + take(b, 0, tr)
    val = take(a, tr, va) + take(b, tr, va)
    test = take(a, tr + va, te) + take(b, tr + va, te)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def test_flex_pajama_ensemble_showcase() -> None:
    global _start
    _start = time.monotonic()
    dspy.configure(lm=EXEC_LM)
    train, val, test = _splits()
    _log(f"PAJAMA ensemble | judges={N_JUDGES} train={len(train)} val={len(val)} test={len(test)} (balanced A/B)")

    # LLM-as-a-judge reference (the single-program baseline), for the plot's comparison line.
    _log("evaluating the LLM-as-judge baseline on the test set (runs the RLM, the slow part)...")
    base_acc, base_calls = _evaluate(dspy.Flex(PairwiseJudge), test)
    _log(f"baseline: accuracy={base_acc:.1%}, avg LLM calls/example={base_calls:.1f}")

    judges = _synthesize_judges(train, val)

    _log("collecting each judge's votes on val + test...")
    gold_test, gold_val = _gold_vec(test), _gold_vec(val)
    test_votes, val_votes = [], []
    for i, judge in enumerate(judges):
        test_votes.append(_judge_votes(judge, test))
        val_votes.append(_judge_votes(judge, val))
        _log(f"  judge {i + 1}/{N_JUDGES}: test accuracy={_accuracy(test_votes[-1], gold_test):.1%}")
    individual = [_accuracy(v, gold_test) for v in test_votes]

    emphases = [JUDGE_EMPHASES[i % len(JUDGE_EMPHASES)] for i in range(N_JUDGES)]
    equal = [1.0] * N_JUDGES
    _log("optimizing the aggregator (EnsembleWeighting) with GEPA...")
    aggregator = _optimize_aggregator(val_votes, gold_val, emphases)
    weights = _learned_weights(aggregator, val_votes, gold_val, emphases)
    _log("aggregator weights: " + ", ".join(f"{w:+.2f}" for w in weights))

    majority_acc = _accuracy(_aggregate(test_votes, equal), gold_test)
    weighted_acc = _accuracy(_aggregate(test_votes, weights), gold_test)
    # Accuracy vs #programs (the paper's Fig. 2), majority vote over the first k judges.
    curve = [_accuracy(_aggregate(test_votes, equal, k=k), gold_test) for k in range(1, N_JUDGES + 1)]
    _log(f"ensemble: majority={majority_acc:.1%}  signature-weighted={weighted_acc:.1%}  (baseline={base_acc:.1%})")
    _log("accuracy vs #programs (majority): " + ", ".join(f"{c:.1%}" for c in curve))

    artifact = {
        "judges": [j.module_src for j in judges],
        "aggregator_instructions": next(p.signature.instructions for _, p in aggregator.named_predictors()),
    }
    SRC_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    _log(f"saved {N_JUDGES} judge sources + aggregator -> {SRC_PATH}")

    xs = list(range(1, N_JUDGES + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, curve, "-o", color="#1a73e8", label="majority-vote ensemble")
    ax.scatter(xs, individual, color="#9aa0a6", zorder=3, label="individual judges")
    ax.scatter([N_JUDGES], [weighted_acc], marker="*", s=160, color="#e37400", zorder=4, label="signature-weighted")
    ax.axhline(base_acc, ls="--", c="#188038", lw=1, label=f"LLM-as-judge ({base_acc:.0%})")
    ax.axhline(0.5, ls=":", c="#c00", lw=1, label="chance")
    ax.set_xlabel("# programs aggregated")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_title(f"PAJAMA multi-program aggregation on JudgeLM (n={len(test)})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    _log(f"saved plot -> {PLOT_PATH}")
    _log("done.")

    # Invariants only: whether aggregation actually climbs depends on the live models/budget, so it's
    # reported and plotted rather than asserted (like the single-program demo).
    assert len(judges) == N_JUDGES
    assert PLOT_PATH.exists()
    assert all(0.0 <= a <= 1.0 for a in [*individual, majority_acc, weighted_acc, *curve])


if __name__ == "__main__":
    test_flex_pajama_ensemble_showcase()
