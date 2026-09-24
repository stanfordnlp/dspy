"""Fit the numeric decision parameters of every Predict in a program against the metric.

Each `Predict` holds per-output parameters in `fields` that reinterpret the backend's
probabilities without adding request parameters. Each candidate reruns the program over the
training set, reusing cached answers for identical requests. Changed upstream decisions can
produce new downstream requests. The search runs directly against the metric, whatever shape
the program's outputs have.

Three kinds of parameter are fitted, all in the `Predict`'s `fields[field]` configuration:
- `threshold`, a Boolean output's cut point on P(True).
- `cuts`, a Score's boundaries on the mean level index. They pick the returned `.level` and leave
  `.value` alone.
- `weights`, a Choice's multiplier per option label, applied to the probabilities before the
  option is picked. An option picked too often gets a multiplier below 1.

The candidate settings come from the training set. One pass records the probabilities the output
is decoded from on every call. A setting anywhere between two neighbouring observed values makes
the same decisions, so the candidates are the midpoints of those gaps: between P(True) values for
a threshold, between mean level indexes for a cut, and between the points where an option's pick
flips for a weight. Among equal scores the candidate in the widest gap wins.

A setting stays unless a candidate scores strictly better and the gain holds across folds. The
fold check splits the training set into FOLDS parts. For each part, it picks a setting on the
other parts and scores that pick on the held-out part. A candidate replaces the current setting
only when those held-out scores beat the current setting's. This discourages gains confined to
small portions of the dataset, but can accept them when they recur across folds.

Calibration enables probability-based execution for every compatible output by adding an entry
in `fields`. After fitting, it compares against the original behavior under the same fold check
and restores the original field configuration unless the fitted behavior wins. The adapter
handles probability requests for the backend used on each call.

"""

import copy
import itertools
import math
import random
from typing import Any, Callable

import dspy
from dspy.adapters.decision import record_evidence
from dspy.adapters.decision_state import DecisionState
from dspy.adapters.types.decision import Choice, Noul, decision_type
from dspy.predict.predict import Predict
from dspy.utils.parallelizer import ParallelExecutor

MAX_CANDIDATES = 40  # candidate settings per parameter; more distinct values are thinned to quantiles
WEIGHT_RANGE = (1e-3, 1e3)  # the smallest and largest Choice multiplier tried
FOLDS = 5  # training-set parts the fold check holds out in turn


def decision_outputs(predict: Predict) -> dict[str, type]:
    """The outputs a decision parameter can apply to, with their decision types."""
    kinds = {name: decision_type(field) for name, field in predict.signature.output_fields.items()}
    return {name: kind for name, kind in kinds.items() if kind is not None}


def predictors(program) -> list[tuple[str, Predict]]:
    """Every Predict with a decision output, by the name `named_parameters` gives it. A bare Predict is `self`."""
    return [(name, p) for name, p in program.named_parameters() if isinstance(p, Predict) and decision_outputs(p)]


def caches(predict: Predict) -> bool:
    """Check caching for a statically bound or configured client."""
    lm = predict.lm or dspy.settings.lm
    if lm is None:
        raise ValueError(
            "ReAnchor cannot check caching without a bound or globally configured client. "
            "For runtime client selection, pass require_cache=False."
        )
    enabled = predict.config.get("cache", getattr(lm, "cache", False))
    return bool(enabled) and getattr(lm, "_cache_responses", True)


def effective(predict: Predict, field: str) -> dict:
    """The output's full configuration: its stored overrides on top of its type's defaults."""
    fields = dict(predict.fields)
    fields.setdefault(field, {})
    return DecisionState(predict.signature, fields).fields[field]


def metric_value(metric: Callable, example, pred) -> float:
    """The metric's score as a float, from a number or from a prediction carrying `score`."""
    verdict = metric(example, pred)
    return verdict.score if hasattr(verdict, "score") else float(verdict)


def scores(
    program, examples: list, metric: Callable, num_threads: int | None = None, progress: bool = False
) -> list[float]:
    """The program's metric score on each example. Any program or metric error fails the pass."""

    def one(example):
        with dspy.context(trace=[]):
            pred = program(**example.inputs())
        return metric_value(metric, example, pred)

    executor = ParallelExecutor(num_threads=num_threads, max_errors=1, disable_progress_bar=not progress)
    values = executor.execute(one, examples)
    if not all(math.isfinite(s) for s in values):
        raise ValueError("ReAnchor requires finite metric values.")
    return values


def run(program, examples: list, metric: Callable, num_threads: int | None = None, progress: bool = False) -> float:
    """The program's mean metric score over `examples`."""
    values = scores(program, examples, metric, num_threads, progress)
    return math.fsum(values) / len(values)


def calibrate(
    program,
    trainset: list,
    metric: Callable,
    num_threads: int | None = None,
) -> list[dict[str, Any]]:
    """Fit every threshold, cut, and weight in place.

    Returns one report row per output: the fitted value, or why it was skipped.
    """
    report = []

    def score() -> list[float]:
        return scores(program, trainset, metric, num_threads)

    for name, predict in predictors(program):
        for field, kind in decision_outputs(predict).items():
            original = copy.deepcopy(predict.fields.get(field))
            before = score()
            predict.fields[field] = effective(predict, field)
            evidence = _observe(program, predict, field, trainset, metric, num_threads)
            row = {"predictor": name, "field": field}
            parameter, fit = (
                ("threshold", _fit_threshold)
                if issubclass(kind, Noul)
                else ("weights", _fit_weights)
                if issubclass(kind, Choice)
                else ("cuts", _fit_cuts)
            )
            base = score()
            best, fitted, observed, check = fit(predict, field, kind, evidence, score, base)
            row.update(parameter=parameter, value=best, train_score=_mean(fitted), train_score_at_start=_mean(base))
            row.update(observed=observed, fold_check=check, train_score_original=_mean(before))
            if _select(before, [(fitted, (), None)])[0] is None:
                _restore(predict, field, original)
                row.pop("value")
                row.update(skipped="fitted behavior did not beat the original", train_score=_mean(before))
            report.append(row)
    return report


def _restore(predict: Predict, field: str, original: dict | None) -> None:
    """Put back the output's entry in `fields` as it was before calibration, or remove it."""
    if original is None:
        del predict.fields[field]
    else:
        predict.fields[field] = original


def _observe(program, predict: Predict, field: str, trainset: list, metric: Callable, num_threads) -> list[dict]:
    """The evidence the output is decoded from on each training call, from one pass at the current setting."""
    with record_evidence() as log:
        scores(program, trainset, metric, num_threads)
    return [evidence for caller, name, evidence in log if caller is predict and name == field]


def _tidy(x: float, lo: float, hi: float) -> float:
    """The shortest decimal strictly between `lo` and `hi`, rounded from `x`."""
    for digits in range(1, 12):
        rounded = round(x, digits)
        if lo < rounded < hi:
            return rounded
    return x


def _gaps(values: list[float], lo: float, hi: float, geometric: bool = False) -> list[tuple[float, float]]:
    """Candidate settings and the width of the gap each sits in.

    The candidates are the midpoints between neighbouring distinct values, with `lo` and `hi`
    closing the range. A setting anywhere inside one gap makes the same decisions as its midpoint,
    so these cover every outcome. More than MAX_CANDIDATES gaps are thinned to evenly spaced
    quantiles of the values. `geometric` takes midpoints and widths on a log scale.
    """
    inner = sorted({v for v in values if lo < v < hi})
    if len(inner) > MAX_CANDIDATES - 1:
        step = (len(inner) - 1) / (MAX_CANDIDATES - 2)
        inner = sorted({inner[round(i * step)] for i in range(MAX_CANDIDATES - 1)})
    points = [lo, *inner, hi]
    gaps = []
    for a, b in itertools.pairwise(points):
        middle, width = (math.sqrt(a * b), math.log(b / a)) if geometric else ((a + b) / 2, b - a)
        if a < middle < b:  # Adjacent floats can round the midpoint onto a boundary.
            gaps.append((_tidy(middle, a, b), width))
    return gaps


def _summary(values: list[float]) -> dict[str, Any]:
    distinct = sorted(set(values))
    if not distinct:
        return {"calls": 0}
    return {
        "calls": len(values),
        "distinct": len(distinct),
        "min": round(distinct[0], 4),
        "max": round(distinct[-1], 4),
    }


def _mean(values: list[float]) -> float:
    return round(math.fsum(values) / len(values), 4)


def _folds(n: int) -> list[list[int]]:
    """Example indexes split into up to FOLDS parts, the same way on every run."""
    order = list(range(n))
    random.Random(0).shuffle(order)
    k = min(FOLDS, n)
    return [order[i::k] for i in range(k)]


def _pick(start: list[float], tried: list[tuple], rows) -> tuple | None:
    """The best entry of `tried` on `rows`, or None when none scores strictly better than `start` there.

    Each entry is (per-example scores, tie-break key, setting). Equal totals go to the larger key.
    """

    def total(values):
        return math.fsum(values[i] for i in rows)

    best = max(tried, key=lambda t: (total(t[0]), t[1]), default=None)
    return best if best is not None and total(best[0]) > total(start) else None


def _select(start: list[float], tried: list[tuple]) -> tuple[tuple | None, bool]:
    """The entry of `tried` to keep, or None to keep the start, and whether the fold check refused a better entry.

    The entry that scores best on the whole training set is kept only when picking on all folds but
    one, and scoring the pick on the held-out fold, beats the start on those same examples.
    """
    best = _pick(start, tried, range(len(start)))
    if best is None:
        return None, False
    held, held_start = 0.0, 0.0
    for fold in _folds(len(start)):
        out = set(fold)
        pick = _pick(start, tried, [i for i in range(len(start)) if i not in out])
        held += math.fsum((pick[0] if pick else start)[i] for i in fold)
        held_start += math.fsum(start[i] for i in fold)
    return (best, False) if held > held_start else (None, True)


def _fit_threshold(
    predict: Predict, field: str, kind: type, evidence: list[dict], score: Callable, base: list[float]
) -> tuple[float, list[float], dict, dict]:
    """The threshold that scores best, between the observed probabilities.

    The starting threshold stays unless a candidate scores strictly better and passes the fold
    check. Among equal scores, the candidate in the widest gap wins, since it leaves the most room
    on either side.
    """
    config = predict.fields[field]
    start = config["threshold"]
    probabilities = [e["noul"] for e in evidence]
    tried = []
    candidates = _gaps(probabilities, 0.0, 1.0)
    if 0.0 in probabilities:
        candidates.append((0.0, 0.0))  # P(True) >= threshold makes zero a distinct outcome.
    for t, width in candidates:
        config["threshold"] = t
        tried.append((score(), (width, -abs(t - start), t), t))
    kept, refused = _select(base, tried)
    best, values = (kept[2], kept[0]) if kept else (start, base)
    config["threshold"] = best
    observed = {**_summary(probabilities), "candidates": len(tried)}
    return best, values, observed, {"passed": int(kept is not None), "failed": int(refused)}


def _fit_weights(
    predict: Predict, field: str, kind: type, evidence: list[dict], score: Callable, base: list[float]
) -> tuple[dict, list[float], dict, dict]:
    """The Choice multipliers that score best on the training set.

    Each option's multiplier moves in turn while the others hold. An option's pick on a call flips
    where its weighted probability meets the strongest rival's, so the candidates sit between those
    flip points. A candidate replaces the current multipliers only when it scores strictly better
    and passes the fold check.
    """
    labels = list(kind.criteria())
    config = predict.fields[field]
    best = {label: config["weights"].get(label, 1.0) for label in labels}
    best_scores, candidates, check = base, 0, {"passed": 0, "failed": 0}
    for label in labels:
        flips = []
        for e in evidence:
            p = e["probabilities"]
            rival = max(best[other] * p[other] for other in labels if other != label)
            if p[label] > 0 and rival > 0:
                flips.append(rival / p[label])
        low, high = WEIGHT_RANGE
        if flips:
            low, high = max(low, min(flips) / 4), min(high, max(flips) * 4)
        tried = []
        for w, width in _gaps(flips, low, high, geometric=True):
            if w == best[label]:
                continue
            candidate = {**best, label: w}
            config["weights"] = candidate
            tried.append((score(), width, candidate))
        candidates += len(tried)
        kept, refused = _select(best_scores, tried)
        check["passed"] += kept is not None
        check["failed"] += refused
        if kept:
            best_scores, _, best = kept
    config["weights"] = best
    return best, best_scores, {"calls": len(evidence), "candidates": candidates}, check


def _fit_cuts(
    predict: Predict, field: str, kind: type, evidence: list[dict], score: Callable, base: list[float]
) -> tuple[list[float], list[float], dict, dict]:
    """The Score cuts that score best on the training set.

    A call's level depends only on its mean level index. Each cut moves in turn between the
    observed means that lie between its neighbours, and stays strictly between them. A candidate
    replaces the current cuts only when it scores strictly better and passes the fold check.
    """
    config = predict.fields[field]
    top = len(kind.criteria()) - 1
    means = [sum(i * p for i, p in e["probabilities"].items()) / sum(e["probabilities"].values()) for e in evidence]
    best, best_scores, candidates, check = list(config["cuts"]), base, 0, {"passed": 0, "failed": 0}
    for i in range(len(best)):
        below = best[i - 1] if i else 0
        above = best[i + 1] if i + 1 < len(best) else top
        tried = []
        for c, width in _gaps(means, below, above):
            if c == best[i]:
                continue
            candidate = [*best[:i], c, *best[i + 1 :]]
            config["cuts"] = candidate
            tried.append((score(), width, candidate))
        candidates += len(tried)
        kept, refused = _select(best_scores, tried)
        check["passed"] += kept is not None
        check["failed"] += refused
        if kept:
            best_scores, _, best = kept
    config["cuts"] = best
    return best, best_scores, {**_summary(means), "candidates": candidates}, check
