"""Fit every Decide's numeric parameters in a program against the metric.

A System One model's answers often track the label and lean. Fixing the lean needs no LM. Each
`Decide` owns parameters that reinterpret the provider's distributions without changing the
request, so cached answers are reused and each candidate setting is one pass of plain Python over
the training set. The search runs directly against the metric, whatever shape the program's
outputs have.

Three kinds of parameter are fitted, all in the `Decide`'s `fields[field]` configuration:
- `threshold`, a Boolean output's cut point on P(True).
- `cuts`, a rich Score's boundaries on the mean level index. They pick the returned `.level`
  and leave `.value` alone. A native `Annotated[float, Score[...]]` output returns only the
  value, so its cuts are not fitted.
- `weights`, a Choice's multiplier per option label, from 0.1 to 10, applied to the probabilities before the
  option is picked. An option picked too often gets a multiplier below 1.

Before sweeping an output, calibration scores it at its current setting and then pushes its
decisions to each extreme: every Noul True, then every Noul False; every Score at its lowest
level, then its highest; each Choice option in turn. When no example's metric score changes
across those settings, the metric does not read the output, and its sweep is skipped. Scores are
compared example by example, because a mean can match by coincidence, as accuracy on balanced
labels does at "all True" and "all False". The current setting catches an example that holds
balanced items of its own, which scores the same at both extremes.
"""

import math
from typing import Any, Callable

import dspy
from dspy.adapters.types.decision import decision_type
from dspy.predict.decide import Decide
from dspy.utils.parallelizer import ParallelExecutor

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
MULTIPLIERS = [0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 6.5, 10.0]  # one Choice option's multiplier
SCORE_STEPS = 10  # a Score's level range is searched in this many steps; each cut is then refined by half a step


def decides(program) -> list[tuple[str, Decide]]:
    """Every Decide in the program, by the name `named_parameters` gives it. A bare Decide is `self`."""
    return [(name, p) for name, p in program.named_parameters() if isinstance(p, Decide)]


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

    `only` limits the fit to the Decide with that name, and `outputs` to those of its outputs.
    `ignored` caches, by (Decide name, output), whether the metric ignores an output. Wording does
    not change what the metric reads, so repeated calls with one cache probe each output once.
    Returns one report row per output: the fitted value, or why it was skipped.
    """
    ignored = {} if ignored is None else ignored
    report = []

    def score() -> float:
        return run(program, trainset, metric, num_threads)

    for name, decide in decides(program):
        if only is not None and name != only:
            continue
        for field, config in decide.fields.items():
            if outputs is not None and field not in outputs:
                continue
            if "cuts" in config and decide.signature.output_fields[field].annotation is float:
                continue  # A native Score returns only the value, which cuts do not move.
            if (name, field) not in ignored:
                ignored[name, field] = _ignored(program, decide, field, trainset, metric, num_threads)
            if ignored[name, field]:
                report.append({"predictor": name, "field": field, "skipped": "the metric does not read this output"})
                continue
            if "threshold" in config:
                # The starting threshold stays unless a grid value scores strictly better.
                start = config["threshold"]
                base = score()
                scores = {}
                for t in THRESHOLDS:
                    config["threshold"] = t
                    scores[t] = base if t == start else score()
                best = max(scores, key=lambda t: (scores[t], -abs(t - 0.5)))
                if scores[best] <= base:
                    best = start
                    scores[start] = base
                config["threshold"] = best
                report.append(
                    {
                        "predictor": name,
                        "field": field,
                        "parameter": "threshold",
                        "value": best,
                        "train_score": round(scores[best], 4),
                        "train_score_at_start": round(base, 4),
                        "train_score_at_default": round(scores[0.5], 4),
                    }
                )
                continue
            base = score()
            if "weights" in config:
                parameter, (best, best_score) = "weights", _fit_multipliers(decide, field, score, base)
            else:
                parameter, (best, best_score) = "cuts", _fit_cuts(decide, field, score, base)
            report.append(
                {
                    "predictor": name,
                    "field": field,
                    "parameter": parameter,
                    "value": best,
                    "train_score": round(best_score, 4),
                    "train_score_at_start": round(base, 4),
                }
            )
    return report


def _extremes(decide: Decide, field: str) -> list[tuple[str, Any]]:
    """Settings that push every decision on an output to one end: (parameter, value) pairs."""
    config = decide.fields[field]
    if "threshold" in config:
        return [("threshold", 0.0), ("threshold", 1.0)]
    options = decision_type(decide.signature.output_fields[field]).options
    if "cuts" in config:
        top = len(options) - 1
        tiny = 1e-6
        return [("cuts", [top - tiny * (top - i) for i in range(top)]), ("cuts", [tiny * (i + 1) for i in range(top)])]
    labels = [str(value) for value, _ in options]
    return [("weights", {other: 1.0 if other == label else 1e-6 for other in labels}) for label in labels]


def _ignored(program, decide: Decide, field: str, trainset: list, metric: Callable, num_threads) -> bool:
    """Whether every example scores the same at the output's current setting and at each extreme."""
    config = decide.fields[field]
    settings = _extremes(decide, field)
    saved = config[settings[0][0]]
    try:
        seen = [scores(program, trainset, metric, num_threads)]
        for parameter, value in settings:
            config[parameter] = value
            seen.append(scores(program, trainset, metric, num_threads))
    finally:
        config[settings[0][0]] = saved
    return all(values == seen[0] for values in seen[1:])


def _fit_multipliers(decide: Decide, field: str, score: Callable, base: float) -> tuple[dict, float]:
    """The Choice multipliers that score best on the training set.

    Each option's multiplier moves in turn over a log-scale grid while the others hold. Ties go to
    the multipliers nearest 1.
    """
    labels = [str(value) for value, _ in decision_type(decide.signature.output_fields[field]).options]
    config = decide.fields[field]
    start = {label: config["weights"].get(label, 1.0) for label in labels}

    def distance(weights):
        return sum(abs(math.log(w)) for w in weights.values())

    best, best_score = start, base
    for label in labels:
        for w in MULTIPLIERS:
            if w == best[label]:
                continue
            candidate = {**best, label: w}
            config["weights"] = candidate
            sc = score()
            if sc > best_score or (sc == best_score and distance(candidate) < distance(best)):
                best, best_score = candidate, sc
    config["weights"] = best
    return best, best_score


def _fit_cuts(decide: Decide, field: str, score: Callable, base: float) -> tuple[list[float], float]:
    """The Score cuts that score best on the training set.

    Each cut moves in turn over a grid strictly inside the level range, then by half a step either
    way, and stays strictly between its neighbours. Ties go to the cuts nearest the defaults,
    halfway between levels, so a metric that ignores `.level` keeps them. A three-level Score
    costs about two dozen passes.
    """
    config = decide.fields[field]
    top = len(decision_type(decide.signature.output_fields[field]).options) - 1
    default = [i + 0.5 for i in range(top)]
    step = top / SCORE_STEPS

    def distance(cuts):
        return sum(abs(c - d) for c, d in zip(cuts, default, strict=True))

    best, best_score = list(config["cuts"]), base
    for i in range(len(best)):
        coarse = [round(step * j, 6) for j in range(1, SCORE_STEPS)]
        for stage in ("coarse", "fine"):
            grid = coarse if stage == "coarse" else [round(best[i] - step / 2, 6), round(best[i] + step / 2, 6)]
            for c in grid:
                below = best[i - 1] if i else 0
                above = best[i + 1] if i + 1 < len(best) else top
                if not below < c < above or c == best[i]:
                    continue
                candidate = [*best[:i], c, *best[i + 1 :]]
                config["cuts"] = candidate
                sc = score()
                if sc > best_score or (sc == best_score and distance(candidate) < distance(best)):
                    best, best_score = candidate, sc
    config["cuts"] = best
    return best, best_score
