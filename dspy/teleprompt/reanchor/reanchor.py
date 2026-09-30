"""Fit a program's decision thresholds, cuts, and weights against the metric."""

import json
import logging
from pathlib import Path
from typing import Any

from dspy.predict.flex.bridge import record_flex_predictors
from dspy.teleprompt.reanchor.calibrate import (
    caches,
    calibrate,
    flex_modules,
    flex_predictors,
    lm_caches,
    predictors,
    run,
)
from dspy.teleprompt.reanchor.decompose import propose, trace
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental
from dspy.utils.exceptions import LMError

logger = logging.getLogger(__name__)


@experimental
class ReAnchor(Teleprompter):
    """Calibrate a program's decisions: fit every threshold, Score cut, and Choice weight against the metric.

    The program can be a single `Predict`, a `dspy.Flex`, or any `dspy.Module` holding predictors
    with decision outputs, and the metric is the only supervision. Each parameter changes how `Predict`
    reads the backend's probabilities without adding request parameters. Identical requests reuse cached
    answers, but changed upstream decisions can produce new downstream requests.

    ReAnchor enables probability-based execution for every compatible output through the predictor's
    `fields`. It keeps each fitted configuration only when it beats the original behavior and passes
    a fold check; otherwise it restores the original configuration, including absent field entries.

    A Flex builds its predictors from its code on every forward, so they are found by running the
    program on the training set, and their fitted parameters are stored in the Flex's `predictor_fields`.
    With a `proposer`, ReAnchor also rewrites each Flex's code: an RLM on the proposer reads the training
    examples, the calibrated outputs, and the probabilities behind every decision, and decomposes the
    judgment into narrower decisions that the code combines. Each rewrite is calibrated in turn and kept
    only when the calibrated program scores higher on the validation set (the training set when there is
    none), or the same with fewer predictor calls per example.

    Args:
        metric: Per-example metric to maximize, as in `dspy.Evaluate`. It may return a number, or a
            `dspy.Prediction` with a `score`.
        num_threads: Evaluation concurrency, as in `dspy.Evaluate`.
        log_dir: When set, report.json is written here.
        require_cache: When True, check caching on statically bound or globally configured clients.
            Set it to False for runtime client selection or to allow uncached requests.
        proposer: A generative LM that rewrites each Flex's code. Without one, only the decision
            parameters are fitted. The Flex's predictors keep running on the Flex's LM or the
            configured one, such as a System One client.
        rounds: Rewrites proposed per Flex when a `proposer` is given.
        proposer_max_iters: REPL steps the proposing RLM may take per rewrite.

    After `compile`, `report` holds the fitted parameters and the metric's mean before and after
    calibration, on the training set and on the validation set when one is given. With a proposer,
    `report["decomposition"]` lists every rewrite with its scores, requests per example, and errors.
    """

    def __init__(
        self,
        metric,
        *,
        num_threads=None,
        log_dir=None,
        require_cache=True,
        proposer=None,
        rounds: int = 4,
        proposer_max_iters: int = 20,
    ):
        super().__init__()
        self.metric = metric
        self.num_threads = num_threads
        self.log_dir = Path(log_dir) if log_dir else None
        self.require_cache = require_cache
        self.proposer = proposer
        self.rounds = rounds
        self.proposer_max_iters = proposer_max_iters
        self.report: dict[str, Any] = {}

    def compile(self, student, *, trainset, valset=None):
        """Return a calibrated copy; leave the student unchanged.

        Args:
            student: A `Predict`, a `dspy.Flex`, or a module holding predictors with decision outputs.
            trainset: Examples the parameters are fitted on, and that a proposer reads.
            valset: Examples scored before and after calibration for the report. Nothing is fitted on
                them; with a proposer, they choose between the calibrated codes.
        """
        if not trainset:
            raise ValueError("trainset must contain at least one example.")
        program = student.deepcopy()
        found = predictors(program)
        flexes = flex_modules(program)
        if not found and not flexes:
            raise ValueError("The student must contain at least one Predict with a decision output, or a dspy.Flex.")
        if self.require_cache:
            self._check_cache([name for name, predict in found if not caches(predict)])
            # A Flex's predictors are only known once it runs; check the client they will use first.
            self._check_cache([name for name, flex in flexes if not lm_caches(flex.lm, {})])
        decomposing = self.proposer is not None and bool(flexes)

        logger.info("answering %d training examples", len(trainset))
        with record_flex_predictors() as built:
            before = self._score(program, trainset, progress=True)
        if flexes:
            found += flex_predictors(program, built)
            if not found and not decomposing:
                raise ValueError(
                    "No predictor with a decision output ran on the training set: the student's Flex code "
                    "called none, and it holds no Predict with one."
                )
            self._check_found_cache(found)
        val_before = self._score(program, valset) if valset else None
        logger.info("fitting thresholds, cuts, and weights")
        fitted = calibrate(program, trainset, self.metric, num_threads=self.num_threads, targets=found)
        self.report = {"train_score_before": before, "train_score": self._score(program, trainset), "fitted": fitted}
        logger.info("calibrated: train %s -> %s", before, self.report["train_score"])
        if valset:
            self.report.update(val_score_before=val_before, val_score=self._score(program, valset))
            logger.info("validation: %s -> %s", val_before, self.report["val_score"])
        if decomposing:
            program = self._decompose(program, trainset, valset)

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "report.json").write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        program._compiled = True
        return program

    def _decompose(self, program, trainset: list, valset: list | None):
        """Propose, calibrate, and select rewrites of each Flex's code; return the best calibrated program."""
        if not valset:
            logger.warning(
                "ReAnchor chooses between rewritten Flex codes on the training set, which the proposer reads; "
                "pass a valset to choose on held-out examples."
            )
        selection = valset or trainset
        best = {"program": program, "fitted": self.report["fitted"], **self._measure(program, selection)}
        attempts = [self._attempt(program, best, accepted=True)]
        for round_ in range(1, self.rounds + 1):
            for path, _ in flex_modules(best["program"]):
                flex = dict(flex_modules(best["program"]))[path]
                logger.info("round %d: proposing code for %s", round_, path)
                records = trace(best["program"], trainset, self.metric, self.num_threads)
                candidate = best["program"].deepcopy()
                attempt = {"round": round_, "flex": path}
                try:
                    attempt["source"] = propose(
                        flex,
                        self.proposer,
                        [self._for_proposer(a) for a in sorted(attempts, key=lambda a: -a.get("score", -1))],
                        records,
                        trainset,
                        self.metric,
                        max_iters=self.proposer_max_iters,
                        num_threads=self.num_threads,
                    )
                    dict(flex_modules(candidate))[path]._bind_code(attempt["source"])
                    outcome = self._calibrate_candidate(candidate, trainset, selection)
                except LMError:
                    raise
                # A proposer that returns no usable source, and code that won't run or breaks under some
                # calibrated setting, all fail the round rather than the run.
                except Exception as e:
                    attempts.append({**attempt, "error": f"{type(e).__name__}: {e}", "accepted": False})
                    logger.info("round %d: the proposal for %s failed: %s", round_, path, e)
                    continue
                if "error" in outcome:
                    attempts.append({**attempt, **outcome, "accepted": False})
                    logger.info("round %d: the proposal for %s failed: %s", round_, path, outcome["error"])
                    continue
                better = outcome["score"] > best["score"] or (
                    outcome["score"] == best["score"] and outcome["calls"] < best["calls"]
                )
                attempts.append(self._attempt(candidate, outcome, accepted=better, round=round_, flex=path))
                logger.info(
                    "round %d: %s scores %s with %s calls per example (best %s with %s)%s",
                    round_, path, outcome["score"], outcome["calls"], best["score"], best["calls"],
                    "; kept" if better else "",
                )
                if better:
                    best = {"program": candidate, **outcome}

        final = best["program"]
        self.report["fitted"] = best["fitted"]
        self.report["train_score"] = self._score(final, trainset)
        if valset:
            self.report["val_score"] = self._score(final, valset)
        self.report["decomposition"] = attempts
        return final

    def _calibrate_candidate(self, candidate, trainset: list, selection: list) -> dict[str, Any]:
        """Find, calibrate, and score a rewritten program, or say why it did not run."""
        with record_flex_predictors() as built:
            records = trace(candidate, trainset, self.metric, self.num_threads)
        errors = [r for r in records if "error" in r]
        if errors:
            shown = "; ".join(f"example {r['index']}: {r['error']}" for r in errors[:3])
            return {"error": f"{len(errors)} of {len(trainset)} training examples failed. {shown}"}
        found = predictors(candidate) + flex_predictors(candidate, built)
        self._check_found_cache(found)
        fitted = calibrate(candidate, trainset, self.metric, num_threads=self.num_threads, targets=found)
        return {"fitted": fitted, **self._measure(candidate, selection)}

    def _measure(self, program, examples: list) -> dict[str, Any]:
        """The program's mean score on `examples` and its bridged predictor calls per example."""
        with record_flex_predictors() as built:
            score = self._score(program, examples)
        return {"score": score, "calls": round(len(built) / len(examples), 4)}

    @staticmethod
    def _attempt(program, outcome: dict, accepted: bool, **where) -> dict[str, Any]:
        return {
            **where,
            "source": {path: flex.module_src for path, flex in flex_modules(program)},
            "score": outcome["score"],
            "calls": outcome["calls"],
            "fitted": outcome["fitted"],
            "accepted": accepted,
        }

    @staticmethod
    def _for_proposer(attempt: dict) -> dict[str, Any]:
        """An attempt as the proposer reads it: the code, how it scored after calibration, or its error."""
        keys = ("source", "score", "calls", "error")
        shown = {k: attempt[k] for k in keys if k in attempt}
        if "fitted" in attempt:
            shown["fitted"] = [{k: v for k, v in row.items() if k != "observed"} for row in attempt["fitted"]]
        return shown

    def _check_found_cache(self, found: list) -> None:
        if self.require_cache:
            self._check_cache([name for name, predict in found if not caches(predict)])

    @staticmethod
    def _check_cache(uncached: list[str]) -> None:
        if uncached:
            raise ValueError(
                f"Predictors {uncached} do not cache responses, so every candidate setting would send new requests. "
                "Enable the client's cache, or pass require_cache=False to calibrate anyway."
            )

    def _score(self, program, examples: list, progress: bool = False) -> float:
        return round(run(program, examples, self.metric, self.num_threads, progress), 4)
