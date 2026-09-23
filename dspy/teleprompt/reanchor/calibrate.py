"""Fit the numeric decision parameters of every Predict in a program against the metric.

Each `Predict` holds per-output parameters in `fields` that reinterpret the backend's
probabilities without changing the request, so cached answers are reused and each candidate
setting is one pass of plain Python over the training set. The search runs directly against the
metric, whatever shape the program's outputs have.

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
flips for a weight. A setting stays unless a candidate scores strictly better, and among equal
scores the candidate in the widest gap wins.

On a generative LM, a native `bool` or `Literal` output returns its value without probabilities
unless it has an entry in `fields`. Calibration adds that entry, which asks the LM for
probabilities, and keeps it only when the fitted setting scores strictly better than the native
output did.

Before sweeping an output, calibration scores it at its current setting and then pushes its
decisions to each extreme: every Noul True, then every Noul False; every Score at its lowest
level, then its highest; each Choice option in turn. When no example's metric score changes
across those settings, the metric does not read the output, and its sweep is skipped. Scores are
compared example by example, because a mean can match by coincidence, as accuracy on balanced
labels does at "all True" and "all False". The current setting catches an example that holds
balanced items of its own, which scores the same at both extremes.
"""

import copy
import itertools
import math
from typing import Any, Callable

import dspy
from dspy.adapters.decision import record_evidence, resolve_adapter
from dspy.adapters.types.decision import Choice, Noul, Score, decision_type
from dspy.predict.predict import Predict
from dspy.utils.parallelizer import ParallelExecutor

MAX_CANDIDATES = 40  # candidate settings per parameter; more distinct values are thinned to quantiles
WEIGHT_RANGE = (1e-3, 1e3)  # the smallest and largest Choice multiplier tried


def decision_outputs(predict: Predict) -> dict[str, type]:
    """The outputs a decision parameter can apply to, with their decision types."""
    kinds = {name: decision_type(field) for name, field in predict.signature.output_fields.items()}
    return {name: kind for name, kind in kinds.items() if kind is not None}


def predictors(program) -> list[tuple[str, Predict]]:
    """Every Predict with a decision output, by the name `named_parameters` gives it. A bare Predict is `self`."""
    return [(name, p) for name, p in program.named_parameters() if isinstance(p, Predict) and decision_outputs(p)]


def resolved_lm(predict: Predict):
    """The client the predictor calls: its own, or the configured one."""
    lm = predict.lm or dspy.settings.lm
    if lm is None:
        raise ValueError("ReAnchor needs an LM or decision client, bound to the predictor or configured globally.")
    return lm


def caches(predict: Predict) -> bool:
    """Whether repeated requests from the predictor are answered from the cache."""
    lm = resolved_lm(predict)
    enabled = predict.config.get("cache", getattr(lm, "cache", False))
    return bool(enabled) and getattr(lm, "_cache_responses", True)


def evidenced(predict: Predict) -> set[str]:
    """The outputs the predictor currently decodes from probabilities."""
    adapter = resolve_adapter(resolved_lm(predict), None, predict.signature, predict.fields)
    return set(adapter.state.fields) if adapter is not None else set()


def effective(predict: Predict, field: str) -> dict:
    """The output's full configuration: its stored overrides on top of its type's defaults."""
    fields = {**predict.fields, field: predict.fields.get(field, {})}
    adapter = resolve_adapter(resolved_lm(predict), None, predict.signature, fields)
    return copy.deepcopy(adapter.state.fields[field])


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
    only: str | None = None,
    outputs: set[str] | None = None,
    ignored: dict[tuple[str, str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Fit every threshold, cut, and weight in place.

    `only` limits the fit to the Predict with that name, and `outputs` to those of its outputs.
    `ignored` caches, by (Predict name, output), whether the metric ignores an output. Wording does
    not change what the metric reads, so repeated calls with one cache probe each output once.
    Returns one report row per output: the fitted value, or why it was skipped.
    """
    ignored = {} if ignored is None else ignored
    report = []

    def score() -> float:
        return run(program, trainset, metric, num_threads)

    for name, predict in predictors(program):
        if only is not None and name != only:
            continue
        for field, kind in decision_outputs(predict).items():
            if outputs is not None and field not in outputs:
                continue
            if ignored.get((name, field)):
                report.append({"predictor": name, "field": field, "skipped": "the metric does not read this output"})
                continue
            original = copy.deepcopy(predict.fields.get(field))
            promoted = field not in evidenced(predict)
            native = score() if promoted else None
            predict.fields[field] = effective(predict, field)
            if (name, field) not in ignored:
                ignored[name, field] = _ignored(program, predict, field, kind, trainset, metric, num_threads)
            if ignored[name, field]:
                _restore(predict, field, original)
                report.append({"predictor": name, "field": field, "skipped": "the metric does not read this output"})
                continue
            evidence = _observe(program, predict, field, trainset, metric, num_threads)
            row = {"predictor": name, "field": field}
            if issubclass(kind, Noul):
                row.update(_fit_threshold(predict, field, evidence, score))
            else:
                base = score()
                fit = _fit_weights if issubclass(kind, Choice) else _fit_cuts
                best, best_score, observed = fit(predict, field, kind, evidence, score, base)
                row.update(parameter="weights" if fit is _fit_weights else "cuts", value=best)
                row.update(train_score=round(best_score, 4), train_score_at_start=round(base, 4), observed=observed)
            if promoted:
                row["train_score_native"] = round(native, 4)
                if row["train_score"] <= row["train_score_native"]:
                    _restore(predict, field, original)
                    row = {
                        "predictor": name,
                        "field": field,
                        "skipped": "probabilities did not beat the native output",
                        "train_score_native": row["train_score_native"],
                        "train_score": row["train_score"],
                    }
                else:
                    row["promoted"] = True
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


def _fit_threshold(predict: Predict, field: str, evidence: list[dict], score: Callable) -> dict[str, Any]:
    """The threshold that scores best, between the observed probabilities.

    The starting threshold stays unless a candidate scores strictly better. Among equal scores, the
    candidate in the widest gap wins, since it leaves the most room on either side.
    """
    config = predict.fields[field]
    start = config["threshold"]
    base = score()
    probabilities = [e["noul"] for e in evidence]
    tried = []
    for t, width in _gaps(probabilities, 0.0, 1.0):
        config["threshold"] = t
        tried.append((score(), width, -abs(t - start), t))
    best_score, _, _, best = max(tried) if tried else (base, 0, 0, start)
    if best_score <= base:
        best, best_score = start, base
    config["threshold"] = best
    return {
        "parameter": "threshold",
        "value": best,
        "train_score": round(best_score, 4),
        "train_score_at_start": round(base, 4),
        "observed": {**_summary(probabilities), "candidates": len(tried)},
    }


def _extremes(kind: type) -> list[tuple[str, Any]]:
    """Settings that push every decision on an output to one end: (parameter, value) pairs."""
    if issubclass(kind, Noul):
        return [("threshold", 0.0), ("threshold", 1.0)]
    if issubclass(kind, Score):
        top = len(kind.options) - 1
        tiny = 1e-6
        return [("cuts", [top - tiny * (top - i) for i in range(top)]), ("cuts", [tiny * (i + 1) for i in range(top)])]
    labels = [str(value) for value, _ in kind.options]
    return [("weights", {other: 1.0 if other == label else 1e-6 for other in labels}) for label in labels]


def _ignored(program, predict: Predict, field: str, kind: type, trainset: list, metric: Callable, num_threads) -> bool:
    """Whether every example scores the same at the output's current setting and at each extreme."""
    config = predict.fields[field]
    settings = _extremes(kind)
    saved = config[settings[0][0]]
    try:
        seen = [scores(program, trainset, metric, num_threads)]
        for parameter, value in settings:
            config[parameter] = value
            seen.append(scores(program, trainset, metric, num_threads))
    finally:
        config[settings[0][0]] = saved
    return all(values == seen[0] for values in seen[1:])


def _fit_weights(
    predict: Predict, field: str, kind: type, evidence: list[dict], score: Callable, base: float
) -> tuple[dict, float, dict]:
    """The Choice multipliers that score best on the training set.

    Each option's multiplier moves in turn while the others hold. An option's pick on a call flips
    where its weighted probability meets the strongest rival's, so the candidates sit between those
    flip points. A candidate replaces the current multipliers only when it scores strictly better.
    """
    labels = [str(value) for value, _ in kind.options]
    config = predict.fields[field]
    best = {label: config["weights"].get(label, 1.0) for label in labels}
    best_score, candidates = base, 0
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
        if tried:
            sc, _, candidate = max(tried, key=lambda t: t[:2])
            if sc > best_score:
                best, best_score = candidate, sc
    config["weights"] = best
    return best, best_score, {"calls": len(evidence), "candidates": candidates}


def _fit_cuts(
    predict: Predict, field: str, kind: type, evidence: list[dict], score: Callable, base: float
) -> tuple[list[float], float, dict]:
    """The Score cuts that score best on the training set.

    A call's level depends only on its mean level index. Each cut moves in turn between the
    observed means that lie between its neighbours, and stays strictly between them. A candidate
    replaces the current cuts only when it scores strictly better.
    """
    config = predict.fields[field]
    top = len(kind.options) - 1
    means = [sum(i * p for i, p in e["probabilities"].items()) / sum(e["probabilities"].values()) for e in evidence]
    best, best_score, candidates = list(config["cuts"]), base, 0
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
        if tried:
            sc, _, candidate = max(tried, key=lambda t: t[:2])
            if sc > best_score:
                best, best_score = candidate, sc
    config["cuts"] = best
    return best, best_score, {**_summary(means), "candidates": candidates}
