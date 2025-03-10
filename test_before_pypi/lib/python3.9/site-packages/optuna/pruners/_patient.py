from __future__ import annotations

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection


@experimental_class("2.8.0")
class PatientPruner(BasePruner):
    """Pruner which wraps another pruner with tolerance.

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
                pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1),
            )
            study.optimize(objective, n_trials=20)

    Args:
        wrapped_pruner:
            Wrapped pruner to perform pruning when :class:`~optuna.pruners.PatientPruner` allows a
            trial to be pruned. If it is :obj:`None`, this pruner is equivalent to
            early-stopping taken the intermediate values in the individual trial.
        patience:
            Pruning is disabled until the objective doesn't improve for
            ``patience`` consecutive steps.
        min_delta:
            Tolerance value to check whether or not the objective improves.
            This value should be non-negative.

    """

    def __init__(
        self, wrapped_pruner: BasePruner | None, patience: int, min_delta: float = 0.0
    ) -> None:
        if patience < 0:
            raise ValueError(f"patience cannot be negative but got {patience}.")

        if min_delta < 0:
            raise ValueError(f"min_delta cannot be negative but got {min_delta}.")

        self._wrapped_pruner = wrapped_pruner
        self._patience = patience
        self._min_delta = min_delta

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        step = trial.last_step
        if step is None:
            return False

        intermediate_values = trial.intermediate_values
        steps = np.asarray(list(intermediate_values.keys()))

        # Do not prune if number of step to determine are insufficient.
        if steps.size <= self._patience + 1:
            return False

        steps.sort()
        # This is the score patience steps ago
        steps_before_patience = steps[: -self._patience - 1]
        scores_before_patience = np.asarray(
            list(intermediate_values[step] for step in steps_before_patience)
        )
        # And these are the scores after that
        steps_after_patience = steps[-self._patience - 1 :]
        scores_after_patience = np.asarray(
            list(intermediate_values[step] for step in steps_after_patience)
        )

        direction = study.direction
        if direction == StudyDirection.MINIMIZE:
            maybe_prune = np.nanmin(scores_before_patience) + self._min_delta < np.nanmin(
                scores_after_patience
            )
        else:
            maybe_prune = np.nanmax(scores_before_patience) - self._min_delta > np.nanmax(
                scores_after_patience
            )

        if maybe_prune:
            if self._wrapped_pruner is not None:
                return self._wrapped_pruner.prune(study, trial)
            else:
                return True
        else:
            return False
