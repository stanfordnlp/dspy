from __future__ import annotations

import abc

from optuna._experimental import experimental_class
from optuna.study.study import Study
from optuna.terminator.erroreval import BaseErrorEvaluator
from optuna.terminator.erroreval import CrossValidationErrorEvaluator
from optuna.terminator.erroreval import StaticErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import BestValueStagnationEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.trial import TrialState


class BaseTerminator(metaclass=abc.ABCMeta):
    """Base class for terminators."""

    @abc.abstractmethod
    def should_terminate(self, study: Study) -> bool:
        pass


@experimental_class("3.2.0")
class Terminator(BaseTerminator):
    """Automatic stopping mechanism for Optuna studies.

    This class implements an automatic stopping mechanism for Optuna studies, aiming to prevent
    unnecessary computation. The study is terminated when the statistical error, e.g.
    cross-validation error, exceeds the room left for optimization.

    For further information about the algorithm, please refer to the following paper:

    - `A. Makarova et al. Automatic termination for hyperparameter optimization.
      <https://proceedings.mlr.press/v188/makarova22a.html>`__

    Args:
        improvement_evaluator:
            An evaluator object for assessing the room left for optimization. Defaults to a
            :class:`~optuna.terminator.improvement.evaluator.RegretBoundEvaluator` object.
        error_evaluator:
            An evaluator for calculating the statistical error, e.g. cross-validation error.
            Defaults to a :class:`~optuna.terminator.CrossValidationErrorEvaluator`
            object.
        min_n_trials:
            The minimum number of trials before termination is considered. Defaults to ``20``.

    Raises:
        ValueError: If ``min_n_trials`` is not a positive integer.

    Example:

        .. testcode::

            import logging
            import sys

            from sklearn.datasets import load_wine
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import KFold

            import optuna
            from optuna.terminator import Terminator
            from optuna.terminator import report_cross_validation_scores


            study = optuna.create_study(direction="maximize")
            terminator = Terminator()
            min_n_trials = 20

            while True:
                trial = study.ask()

                X, y = load_wine(return_X_y=True)

                clf = RandomForestClassifier(
                    max_depth=trial.suggest_int("max_depth", 2, 32),
                    min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
                    criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
                )

                scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
                report_cross_validation_scores(trial, scores)

                value = scores.mean()
                logging.info(f"Trial #{trial.number} finished with value {value}.")
                study.tell(trial, value)

                if trial.number > min_n_trials and terminator.should_terminate(study):
                    logging.info("Terminated by Optuna Terminator!")
                    break

    .. seealso::
        Please refer to :class:`~optuna.terminator.TerminatorCallback` for how to use
        the terminator mechanism with the :func:`~optuna.study.Study.optimize` method.

    """

    def __init__(
        self,
        improvement_evaluator: BaseImprovementEvaluator | None = None,
        error_evaluator: BaseErrorEvaluator | None = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        if min_n_trials <= 0:
            raise ValueError("`min_n_trials` is expected to be a positive integer.")

        self._improvement_evaluator = improvement_evaluator or RegretBoundEvaluator()
        self._error_evaluator = error_evaluator or self._initialize_error_evaluator()
        self._min_n_trials = min_n_trials

    def _initialize_error_evaluator(self) -> BaseErrorEvaluator:
        if isinstance(self._improvement_evaluator, BestValueStagnationEvaluator):
            return StaticErrorEvaluator(constant=0)
        return CrossValidationErrorEvaluator()

    def should_terminate(self, study: Study) -> bool:
        """Judge whether the study should be terminated based on the reported values."""
        trials = study.get_trials(states=[TrialState.COMPLETE])

        if len(trials) < self._min_n_trials:
            return False

        improvement = self._improvement_evaluator.evaluate(
            trials=study.trials,
            study_direction=study.direction,
        )

        error = self._error_evaluator.evaluate(
            trials=study.trials, study_direction=study.direction
        )

        should_terminate = improvement < error
        return should_terminate
