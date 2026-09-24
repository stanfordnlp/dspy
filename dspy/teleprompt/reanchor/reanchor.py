"""Fit a program's decision thresholds, cuts, and weights against the metric."""

import json
import logging
from pathlib import Path
from typing import Any

from dspy.teleprompt.reanchor.calibrate import caches, calibrate, predictors, run
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)


@experimental
class ReAnchor(Teleprompter):
    """Calibrate a program's decisions: fit every threshold, Score cut, and Choice weight against the metric.

    The program can be a single `Predict` or any `dspy.Module` holding predictors with decision
    outputs, and the metric is the only supervision. Each parameter changes how `Predict` reads the
    backend's probabilities and leaves the request unchanged. Calibration answers the training set
    once, then searches each parameter against the metric on cached answers.

    On a generative LM, a native `bool` or `Literal` output is answered without probabilities.
    ReAnchor gives such an output an entry in the predictor's `fields`, which asks the LM for
    probabilities, and keeps the entry only when the fitted setting beats the native output on the
    training set.

    Args:
        metric: Per-example metric to maximize, as in `dspy.Evaluate`. It may return a number, or a
            `dspy.Prediction` with a `score`.
        num_threads: Evaluation concurrency, as in `dspy.Evaluate`.
        log_dir: When set, report.json is written here.
        require_cache: When True, `compile` raises if a predictor's client does not cache responses.
            Set it to False to calibrate anyway; every candidate setting then sends new requests.

    After `compile`, `report` holds the fitted parameters and the metric's mean before and after
    calibration, on the training set and on the validation set when one is given.
    """

    def __init__(self, metric, *, num_threads=None, log_dir=None, require_cache=True):
        super().__init__()
        self.metric = metric
        self.num_threads = num_threads
        self.log_dir = Path(log_dir) if log_dir else None
        self.require_cache = require_cache
        self.report: dict[str, Any] = {}

    def compile(self, student, *, trainset, valset=None):
        """Return a calibrated copy; leave the student unchanged.

        Args:
            student: A `Predict`, or a module holding predictors with decision outputs.
            trainset: Examples the parameters are fitted on.
            valset: Examples scored before and after calibration for the report. Nothing is fitted on them.
        """
        if not trainset:
            raise ValueError("trainset must contain at least one example.")
        program = student.deepcopy()
        found = predictors(program)
        if not found:
            raise ValueError("The student must contain at least one Predict with a decision output.")
        uncached = [name for name, predict in found if not caches(predict)]
        if self.require_cache and uncached:
            raise ValueError(
                f"Predictors {uncached} do not cache responses, so every candidate setting would send new requests. "
                "Enable the client's cache, or pass require_cache=False to calibrate anyway."
            )

        logger.info("answering %d training examples", len(trainset))
        before = self._score(program, trainset, progress=True)
        val_before = self._score(program, valset) if valset else None
        logger.info("fitting thresholds, cuts, and weights")
        fitted = calibrate(program, trainset, self.metric, num_threads=self.num_threads)
        self.report = {"train_score_before": before, "train_score": self._score(program, trainset), "fitted": fitted}
        logger.info("calibrated: train %s -> %s", before, self.report["train_score"])
        if valset:
            self.report.update(val_score_before=val_before, val_score=self._score(program, valset))
            logger.info("validation: %s -> %s", val_before, self.report["val_score"])

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "report.json").write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        program._compiled = True
        return program

    def _score(self, program, examples: list, progress: bool = False) -> float:
        return round(run(program, examples, self.metric, self.num_threads, progress), 4)
