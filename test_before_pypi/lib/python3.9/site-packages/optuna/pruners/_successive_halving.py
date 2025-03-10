from __future__ import annotations

import math

import optuna
from optuna.pruners._base import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState


class SuccessiveHalvingPruner(BasePruner):
    """Pruner using Asynchronous Successive Halving Algorithm.

    `Successive Halving <https://proceedings.mlr.press/v51/jamieson16.html>`__ is a bandit-based
    algorithm to identify the best one among multiple configurations. This class implements an
    asynchronous version of Successive Halving. Please refer to the paper of
    `Asynchronous Successive Halving <https://proceedings.mlsys.org/paper_files/paper/2020/file/
    a06f20b349c6cf09a6b171c71b88bbfc-Paper.pdf>`__ for detailed descriptions.

    Note that, this class does not take care of the parameter for the maximum
    resource, referred to as :math:`R` in the paper. The maximum resource allocated to a trial is
    typically limited inside the objective function (e.g., ``step`` number in `simple_pruning.py
    <https://github.com/optuna/optuna-examples/blob/main/basic/pruning.py>`__,
    ``EPOCH`` number in `chainer_integration.py
    <https://github.com/optuna/optuna-examples/tree/main/chainer/chainer_integration.py#L73>`__).

    .. seealso::
        Please refer to :meth:`~optuna.trial.Trial.report`.

    Example:

        We minimize an objective function with ``SuccessiveHalvingPruner``.

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
                direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner()
            )
            study.optimize(objective, n_trials=20)

    Args:
        min_resource:
            A parameter for specifying the minimum resource allocated to a trial
            (in the `paper <https://proceedings.mlsys.org/paper_files/paper/2020/file/
            a06f20b349c6cf09a6b171c71b88bbfc-Paper.pdf>`__ this parameter is referred to as
            :math:`r`).
            This parameter defaults to 'auto' where the value is determined based on a heuristic
            that looks at the number of required steps for the first trial to complete.

            A trial is never pruned until it executes
            :math:`\\mathsf{min}\\_\\mathsf{resource} \\times
            \\mathsf{reduction}\\_\\mathsf{factor}^{
            \\mathsf{min}\\_\\mathsf{early}\\_\\mathsf{stopping}\\_\\mathsf{rate}}`
            steps (i.e., the completion point of the first rung). When the trial completes
            the first rung, it will be promoted to the next rung only
            if the value of the trial is placed in the top
            :math:`{1 \\over \\mathsf{reduction}\\_\\mathsf{factor}}` fraction of
            the all trials that already have reached the point (otherwise it will be pruned there).
            If the trial won the competition, it runs until the next completion point (i.e.,
            :math:`\\mathsf{min}\\_\\mathsf{resource} \\times
            \\mathsf{reduction}\\_\\mathsf{factor}^{
            (\\mathsf{min}\\_\\mathsf{early}\\_\\mathsf{stopping}\\_\\mathsf{rate}
            + \\mathsf{rung})}` steps)
            and repeats the same procedure.

            .. note::
                If the step of the last intermediate value may change with each trial, please
                manually specify the minimum possible step to ``min_resource``.
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials
            (in the `paper <https://proceedings.mlsys.org/paper_files/paper/2020/file/
            a06f20b349c6cf09a6b171c71b88bbfc-Paper.pdf>`__ this parameter is
            referred to as :math:`\\eta`).  At the completion point of each rung,
            about :math:`{1 \\over \\mathsf{reduction}\\_\\mathsf{factor}}`
            trials will be promoted.
        min_early_stopping_rate:
            A parameter for specifying the minimum early-stopping rate
            (in the `paper <https://proceedings.mlsys.org/paper_files/paper/2020/file/
            a06f20b349c6cf09a6b171c71b88bbfc-Paper.pdf>`__ this parameter is
            referred to as :math:`s`).
        bootstrap_count:
            Minimum number of trials that need to complete a rung before any trial
            is considered for promotion into the next rung.
    """

    def __init__(
        self,
        min_resource: str | int = "auto",
        reduction_factor: int = 4,
        min_early_stopping_rate: int = 0,
        bootstrap_count: int = 0,
    ) -> None:
        if isinstance(min_resource, str) and min_resource != "auto":
            raise ValueError(
                "The value of `min_resource` is {}, "
                "but must be either `min_resource` >= 1 or 'auto'".format(min_resource)
            )

        if isinstance(min_resource, int) and min_resource < 1:
            raise ValueError(
                "The value of `min_resource` is {}, "
                "but must be either `min_resource >= 1` or 'auto'".format(min_resource)
            )

        if reduction_factor < 2:
            raise ValueError(
                "The value of `reduction_factor` is {}, "
                "but must be `reduction_factor >= 2`".format(reduction_factor)
            )

        if min_early_stopping_rate < 0:
            raise ValueError(
                "The value of `min_early_stopping_rate` is {}, "
                "but must be `min_early_stopping_rate >= 0`".format(min_early_stopping_rate)
            )

        if bootstrap_count < 0:
            raise ValueError(
                "The value of `bootstrap_count` is {}, "
                "but must be `bootstrap_count >= 0`".format(bootstrap_count)
            )

        if bootstrap_count > 0 and min_resource == "auto":
            raise ValueError(
                "bootstrap_count > 0 and min_resource == 'auto' "
                "are mutually incompatible, bootstrap_count is {}".format(bootstrap_count)
            )

        self._min_resource: int | None = None
        if isinstance(min_resource, int):
            self._min_resource = min_resource
        self._reduction_factor = reduction_factor
        self._min_early_stopping_rate = min_early_stopping_rate
        self._bootstrap_count = bootstrap_count

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        step = trial.last_step
        if step is None:
            return False

        rung = _get_current_rung(trial)
        value = trial.intermediate_values[step]
        trials: list["optuna.trial.FrozenTrial"] | None = None

        while True:
            if self._min_resource is None:
                if trials is None:
                    trials = study.get_trials(deepcopy=False)
                self._min_resource = _estimate_min_resource(trials)
                if self._min_resource is None:
                    return False

            assert self._min_resource is not None
            rung_promotion_step = self._min_resource * (
                self._reduction_factor ** (self._min_early_stopping_rate + rung)
            )
            if step < rung_promotion_step:
                return False

            if math.isnan(value):
                return True

            if trials is None:
                trials = study.get_trials(deepcopy=False)

            rung_key = _completed_rung_key(rung)

            study._storage.set_trial_system_attr(trial._trial_id, rung_key, value)

            competing = _get_competing_values(trials, value, rung_key)

            # 'competing' already includes the current trial
            # Therefore, we need to use the '<=' operator here
            if len(competing) <= self._bootstrap_count:
                return True

            if not _is_trial_promotable_to_next_rung(
                value,
                competing,
                self._reduction_factor,
                study.direction,
            ):
                return True

            rung += 1


def _estimate_min_resource(trials: list["optuna.trial.FrozenTrial"]) -> int | None:
    n_steps = [
        t.last_step for t in trials if t.state == TrialState.COMPLETE and t.last_step is not None
    ]

    if not n_steps:
        return None

    # Get the maximum number of steps and divide it by 100.
    last_step = max(n_steps)
    return max(last_step // 100, 1)


def _get_current_rung(trial: "optuna.trial.FrozenTrial") -> int:
    # The following loop takes `O(log step)` iterations.
    rung = 0
    while _completed_rung_key(rung) in trial.system_attrs:
        rung += 1
    return rung


def _completed_rung_key(rung: int) -> str:
    return "completed_rung_{}".format(rung)


def _get_competing_values(
    trials: list["optuna.trial.FrozenTrial"], value: float, rung_key: str
) -> list[float]:
    competing_values = [t.system_attrs[rung_key] for t in trials if rung_key in t.system_attrs]
    competing_values.append(value)
    return competing_values


def _is_trial_promotable_to_next_rung(
    value: float,
    competing_values: list[float],
    reduction_factor: int,
    study_direction: StudyDirection,
) -> bool:
    promotable_idx = (len(competing_values) // reduction_factor) - 1

    if promotable_idx == -1:
        # Optuna does not support suspending or resuming ongoing trials. Therefore, for the first
        # `eta - 1` trials, this implementation instead promotes the trial if its value is the
        # smallest one among the competing values.
        promotable_idx = 0

    competing_values.sort()
    if study_direction == StudyDirection.MAXIMIZE:
        return value >= competing_values[-(promotable_idx + 1)]
    return value <= competing_values[promotable_idx]
