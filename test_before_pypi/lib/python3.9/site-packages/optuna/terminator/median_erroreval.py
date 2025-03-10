from __future__ import annotations

import sys

import numpy as np

from optuna._experimental import experimental_class
from optuna.study import StudyDirection
from optuna.terminator.erroreval import BaseErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.trial import FrozenTrial
from optuna.trial._state import TrialState


@experimental_class("4.0.0")
class MedianErrorEvaluator(BaseErrorEvaluator):
    """An error evaluator that returns the ratio to initial median.

    This error evaluator is introduced as a heuristics in the following paper:

    - `A stopping criterion for Bayesian optimization by the gap of expected minimum simple
      regrets <https://proceedings.mlr.press/v206/ishibashi23a.html>`__

    Args:
        paired_improvement_evaluator:
            The ``improvement_evaluator`` instance which is set with this ``error_evaluator``.
        warm_up_trials:
            A parameter specifies the number of initial trials to be discarded before
            the calculation of median. Default to 10.
            In optuna, the first 10 trials are often random sampling.
            The ``warm_up_trials`` can exclude them from the calculation.
        n_initial_trials:
            A parameter specifies the number of initial trials considered in the calculation of
            median after `warm_up_trials`. Default to 20.
        threshold_ratio:
            A parameter specifies the ratio between the threshold and initial median.
            Default to 0.01.
    """

    def __init__(
        self,
        paired_improvement_evaluator: BaseImprovementEvaluator,
        warm_up_trials: int = 10,
        n_initial_trials: int = 20,
        threshold_ratio: float = 0.01,
    ) -> None:
        if warm_up_trials < 0:
            raise ValueError("`warm_up_trials` is expected to be a non-negative integer.")
        if n_initial_trials <= 0:
            raise ValueError("`n_initial_trials` is expected to be a positive integer.")
        if threshold_ratio <= 0.0 or not np.isfinite(threshold_ratio):
            raise ValueError("`threshold_ratio_to_initial_median` is expected to be a positive.")

        self._paired_improvement_evaluator = paired_improvement_evaluator
        self._warm_up_trials = warm_up_trials
        self._n_initial_trials = n_initial_trials
        self._threshold_ratio = threshold_ratio
        self._threshold: float | None = None

    def evaluate(
        self,
        trials: list[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:

        if self._threshold is not None:
            return self._threshold

        trials = [trial for trial in trials if trial.state == TrialState.COMPLETE]
        if len(trials) < (self._warm_up_trials + self._n_initial_trials):
            return (
                -sys.float_info.min
            )  # Do not terminate. It assumes that improvement must non-negative.
        trials.sort(key=lambda trial: trial.number)
        criteria = []
        for i in range(1, self._n_initial_trials + 1):
            criteria.append(
                self._paired_improvement_evaluator.evaluate(
                    trials[self._warm_up_trials : self._warm_up_trials + i], study_direction
                )
            )
        criteria.sort()
        self._threshold = criteria[len(criteria) // 2]
        assert self._threshold is not None
        self._threshold = min(sys.float_info.max, self._threshold * self._threshold_ratio)
        return self._threshold
