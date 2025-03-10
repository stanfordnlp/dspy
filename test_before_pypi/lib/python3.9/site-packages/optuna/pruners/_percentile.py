from __future__ import annotations

from collections.abc import KeysView
import functools
import math

import numpy as np

import optuna
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState


def _get_best_intermediate_result_over_steps(
    trial: "optuna.trial.FrozenTrial", direction: StudyDirection
) -> float:
    values = np.asarray(list(trial.intermediate_values.values()), dtype=float)
    if direction == StudyDirection.MAXIMIZE:
        return np.nanmax(values)
    return np.nanmin(values)


def _get_percentile_intermediate_result_over_trials(
    completed_trials: list["optuna.trial.FrozenTrial"],
    direction: StudyDirection,
    step: int,
    percentile: float,
    n_min_trials: int,
) -> float:
    if len(completed_trials) == 0:
        raise ValueError("No trials have been completed.")

    intermediate_values = [
        t.intermediate_values[step] for t in completed_trials if step in t.intermediate_values
    ]

    if len(intermediate_values) < n_min_trials:
        return math.nan

    if direction == StudyDirection.MAXIMIZE:
        percentile = 100 - percentile

    return float(
        np.nanpercentile(
            np.array(intermediate_values, dtype=float),
            percentile,
        )
    )


def _is_first_in_interval_step(
    step: int, intermediate_steps: KeysView[int], n_warmup_steps: int, interval_steps: int
) -> bool:
    nearest_lower_pruning_step = (
        step - n_warmup_steps
    ) // interval_steps * interval_steps + n_warmup_steps
    assert nearest_lower_pruning_step >= 0

    # `intermediate_steps` may not be sorted so we must go through all elements.
    second_last_step = functools.reduce(
        lambda second_last_step, s: s if s > second_last_step and s != step else second_last_step,
        intermediate_steps,
        -1,
    )

    return second_last_step < nearest_lower_pruning_step


class PercentilePruner(BasePruner):
    """Pruner to keep the specified percentile of the trials.

    Prune if the best intermediate value is in the bottom percentile among trials at the same step.

    Example:

        .. testcode::

            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            import optuna

            X, y = load_iris(return_X_y=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)
            classes = np.unique(y)


            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)
                n_train_iter = 100

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_valid, y_valid)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return clf.score(X_valid, y_valid)


            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.PercentilePruner(
                    25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10
                ),
            )
            study.optimize(objective, n_trials=20)

    Args:
        percentile:
            Percentile which must be between 0 and 100 inclusive
            (e.g., When given 25.0, top of 25th percentile trials are kept).
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial exceeds the given number of step. Note that
            this feature assumes that ``step`` starts at zero.
        interval_steps:
            Interval in number of steps between the pruning checks, offset by the warmup steps.
            If no value has been reported at the time of a pruning check, that particular check
            will be postponed until a value is reported. Value must be at least 1.
        n_min_trials:
            Minimum number of reported trial results at a step to judge whether to prune.
            If the number of reported intermediate values from all trials at the current step
            is less than ``n_min_trials``, the trial will not be pruned. This can be used to ensure
            that a minimum number of trials are run to completion without being pruned.
    """

    def __init__(
        self,
        percentile: float,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        if not 0.0 <= percentile <= 100:
            raise ValueError(
                "Percentile must be between 0 and 100 inclusive but got {}.".format(percentile)
            )
        if n_startup_trials < 0:
            raise ValueError(
                "Number of startup trials cannot be negative but got {}.".format(n_startup_trials)
            )
        if n_warmup_steps < 0:
            raise ValueError(
                "Number of warmup steps cannot be negative but got {}.".format(n_warmup_steps)
            )
        if interval_steps < 1:
            raise ValueError(
                "Pruning interval steps must be at least 1 but got {}.".format(interval_steps)
            )
        if n_min_trials < 1:
            raise ValueError(
                "Number of trials for pruning must be at least 1 but got {}.".format(n_min_trials)
            )

        self._percentile = percentile
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps
        self._n_min_trials = n_min_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        n_trials = len(completed_trials)

        if n_trials == 0:
            return False

        if n_trials < self._n_startup_trials:
            return False

        step = trial.last_step
        if step is None:
            return False

        n_warmup_steps = self._n_warmup_steps
        if step < n_warmup_steps:
            return False

        if not _is_first_in_interval_step(
            step, trial.intermediate_values.keys(), n_warmup_steps, self._interval_steps
        ):
            return False

        direction = study.direction
        best_intermediate_result = _get_best_intermediate_result_over_steps(trial, direction)
        if math.isnan(best_intermediate_result):
            return True

        p = _get_percentile_intermediate_result_over_trials(
            completed_trials, direction, step, self._percentile, self._n_min_trials
        )
        if math.isnan(p):
            return False

        if direction == StudyDirection.MAXIMIZE:
            return best_intermediate_result < p
        return best_intermediate_result > p
