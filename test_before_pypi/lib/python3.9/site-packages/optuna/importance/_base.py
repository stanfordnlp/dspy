from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Collection
from typing import cast

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.search_space import intersection_search_space
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class BaseImportanceEvaluator(abc.ABC):
    """Abstract parameter importance evaluator."""

    @abc.abstractmethod
    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        """Evaluate parameter importances based on completed trials in the given study.

        .. note::

            This method is not meant to be called by library users.

        .. seealso::

            Please refer to :func:`~optuna.importance.get_param_importances` for how a concrete
            evaluator should implement this method.

        Args:
            study:
                An optimized study.
            params:
                A list of names of parameters to assess.
                If :obj:`None`, all parameters that are present in all of the completed trials are
                assessed.
            target:
                A function to specify the value to evaluate importances.
                If it is :obj:`None` and ``study`` is being used for single-objective optimization,
                the objective values are used. Can also be used for other trial attributes, such as
                the duration, like ``target=lambda t: t.duration.total_seconds()``.

                .. note::
                    Specify this argument if ``study`` is being used for multi-objective
                    optimization. For example, to get the hyperparameter importance of the first
                    objective, use ``target=lambda t: t.values[0]`` for the target parameter.

        Returns:
            A :obj:`dict` where the keys are parameter names and the values are assessed
            importances.

        """
        # TODO(hvy): Reconsider the interface as logic might violate DRY among multiple evaluators.
        raise NotImplementedError


def _get_distributions(study: Study, params: list[str] | None) -> dict[str, BaseDistribution]:
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    _check_evaluate_args(completed_trials, params)

    if params is None:
        return intersection_search_space(study.get_trials(deepcopy=False))

    # New temporary required to pass mypy. Seems like a bug.
    params_not_none = params
    assert params_not_none is not None

    # Compute the search space based on the subset of trials containing all parameters.
    distributions = None
    for trial in completed_trials:
        trial_distributions = trial.distributions
        if not all(name in trial_distributions for name in params_not_none):
            continue

        if distributions is None:
            distributions = dict(
                filter(
                    lambda name_and_distribution: name_and_distribution[0] in params_not_none,
                    trial_distributions.items(),
                )
            )
            continue

        if any(
            trial_distributions[name] != distribution
            for name, distribution in distributions.items()
        ):
            raise ValueError(
                "Parameters importances cannot be assessed with dynamic search spaces if "
                "parameters are specified. Specified parameters: {}.".format(params)
            )

    assert distributions is not None  # Required to pass mypy.
    distributions = dict(
        sorted(distributions.items(), key=lambda name_and_distribution: name_and_distribution[0])
    )
    return distributions


def _check_evaluate_args(completed_trials: list[FrozenTrial], params: list[str] | None) -> None:
    if len(completed_trials) == 0:
        raise ValueError("Cannot evaluate parameter importances without completed trials.")
    if len(completed_trials) == 1:
        raise ValueError("Cannot evaluate parameter importances with only a single trial.")

    if params is not None:
        if not isinstance(params, (list, tuple)):
            raise TypeError(
                "Parameters must be specified as a list. Actual parameters: {}.".format(params)
            )
        if any(not isinstance(p, str) for p in params):
            raise TypeError(
                "Parameters must be specified by their names with strings. Actual parameters: "
                "{}.".format(params)
            )

        if len(params) > 0:
            at_least_one_trial = False
            for trial in completed_trials:
                if all(p in trial.distributions for p in params):
                    at_least_one_trial = True
                    break
            if not at_least_one_trial:
                raise ValueError(
                    "Study must contain completed trials with all specified parameters. "
                    "Specified parameters: {}.".format(params)
                )


def _get_filtered_trials(
    study: Study, params: Collection[str], target: Callable[[FrozenTrial], float] | None
) -> list[FrozenTrial]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return [
        trial
        for trial in trials
        if set(params) <= set(trial.params)
        and np.isfinite(target(trial) if target is not None else cast(float, trial.value))
    ]


def _param_importances_to_dict(
    params: Collection[str], param_importances: np.ndarray | float
) -> dict[str, float]:
    return {
        name: value
        for name, value in zip(params, np.broadcast_to(param_importances, (len(params),)))
    }


def _get_trans_params(trials: list[FrozenTrial], trans: _SearchSpaceTransform) -> np.ndarray:
    return np.array([trans.transform(trial.params) for trial in trials])


def _get_target_values(
    trials: list[FrozenTrial], target: Callable[[FrozenTrial], float] | None
) -> np.ndarray:
    return np.array([target(trial) if target is not None else trial.value for trial in trials])


def _sort_dict_by_importance(param_importances: dict[str, float]) -> dict[str, float]:
    return dict(
        reversed(
            sorted(
                param_importances.items(), key=lambda name_and_importance: name_and_importance[1]
            )
        )
    )
