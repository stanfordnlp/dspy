"""Fit a Decide program's thresholds, cuts, and weights against the metric."""

import json
import logging
import pprint
from pathlib import Path
from typing import Any

from dspy.teleprompt.reanchor.calibrate import calibrate, decides, run
from dspy.teleprompt.reanchor.source import render_signature
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)


@experimental
class ReAnchor(Teleprompter):
    """Calibrate a Decide program: fit every threshold, rich Score cut, and Choice weight against the metric.

    The program can be a single `Decide` or any `dspy.Module` holding `Decide` parameters, and the
    metric is the only supervision. Each parameter changes how `Decide` reads the System One model's
    probabilities and leaves the request unchanged. Calibration answers the training set once, then
    searches each parameter against the metric on cached answers.

    Args:
        metric: Per-example metric to maximize, as in `dspy.Evaluate`. It may return a number, or a
            `dspy.Prediction` with a `score`.
        num_threads: Evaluation concurrency, as in `dspy.Evaluate`.
        log_dir: When set, the report and the program's source are written here.

    After `compile`, `report` holds the fitted parameters and the metric's mean before and after
    calibration, on the training set and on the validation set when one is given.
    """

    def __init__(self, metric, *, num_threads=None, log_dir=None):
        super().__init__()
        self.metric = metric
        self.num_threads = num_threads
        self.log_dir = Path(log_dir) if log_dir else None
        self.report: dict[str, Any] = {}

    def compile(self, student, *, trainset, teacher=None, valset=None, **kwargs):
        """Return a calibrated copy; leave the student unchanged.

        Args:
            student: A `Decide`, or a module holding `Decide` parameters.
            trainset: Examples the parameters are fitted on.
            valset: Examples scored before and after calibration for the report. Nothing is fitted on them.
        """
        if not trainset:
            raise ValueError("trainset must contain at least one example.")
        program = student.deepcopy()
        if not decides(program):
            raise ValueError("The student must contain at least one discoverable Decide parameter.")

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
            (self.log_dir / "source.py").write_text(self.source(program), encoding="utf-8")
        program._compiled = True
        return program

    def _score(self, program, examples: list, progress: bool = False) -> float:
        return round(run(program, examples, self.metric, self.num_threads, progress), 4)

    @staticmethod
    def source(program) -> str:
        """Every Decide in the program as its declared signature class, followed by the line that sets its `fields`."""
        parts = []
        for name, decide in decides(program):
            target = "program" if name == "self" else f"program.{name}"
            fields = pprint.pformat(decide.fields, width=100, sort_dicts=False)
            parts.append(f"{render_signature(decide.signature)}\n{target}.fields = {fields}\n")
        return "\n\n".join(parts)
