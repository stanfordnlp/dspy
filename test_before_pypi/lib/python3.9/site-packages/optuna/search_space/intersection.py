from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import optuna
from optuna.distributions import BaseDistribution


if TYPE_CHECKING:
    from optuna.study import Study


def _calculate(
    trials: list[optuna.trial.FrozenTrial],
    include_pruned: bool = False,
    search_space: dict[str, BaseDistribution] | None = None,
    cached_trial_number: int = -1,
) -> tuple[dict[str, BaseDistribution] | None, int]:
    states_of_interest = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.WAITING,
        optuna.trial.TrialState.RUNNING,
    ]

    if include_pruned:
        states_of_interest.append(optuna.trial.TrialState.PRUNED)

    trials_of_interest = [trial for trial in trials if trial.state in states_of_interest]

    next_cached_trial_number = (
        trials_of_interest[-1].number + 1 if len(trials_of_interest) > 0 else -1
    )
    for trial in reversed(trials_of_interest):
        if cached_trial_number > trial.number:
            break

        if not trial.state.is_finished():
            next_cached_trial_number = trial.number
            continue

        if search_space is None:
            search_space = copy.copy(trial.distributions)
            continue

        search_space = {
            name: distribution
            for name, distribution in search_space.items()
            if trial.distributions.get(name) == distribution
        }

    return search_space, next_cached_trial_number


class IntersectionSearchSpace:
    """A class to calculate the intersection search space of a :class:`~optuna.study.Study`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Note that an instance of this class is supposed to be used for only one study.
    If different studies are passed to
    :func:`~optuna.search_space.IntersectionSearchSpace.calculate`,
    a :obj:`ValueError` is raised.

    Args:
        include_pruned:
            Whether pruned trials should be included in the search space.
    """

    def __init__(self, include_pruned: bool = False) -> None:
        self._cached_trial_number: int = -1
        self._search_space: dict[str, BaseDistribution] | None = None
        self._study_id: int | None = None

        self._include_pruned = include_pruned

    def calculate(self, study: Study) -> dict[str, BaseDistribution]:
        """Returns the intersection search space of the :class:`~optuna.study.Study`.

        Args:
            study:
                A study with completed trials. The same study must be passed for one instance
                of this class through its lifetime.

        Returns:
            A dictionary containing the parameter names and parameter's distributions sorted by
            parameter names.
        """

        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when
            # :class:`~optuna.storages.InMemoryStorage` is used because
            # :func:`~optuna.storages.InMemoryStorage.create_new_study`
            # always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`IntersectionSearchSpace` cannot handle multiple studies.")

        self._search_space, self._cached_trial_number = _calculate(
            study.get_trials(deepcopy=False),
            self._include_pruned,
            self._search_space,
            self._cached_trial_number,
        )
        search_space = self._search_space or {}
        search_space = dict(sorted(search_space.items(), key=lambda x: x[0]))
        return copy.deepcopy(search_space)


def intersection_search_space(
    trials: list[optuna.trial.FrozenTrial],
    include_pruned: bool = False,
) -> dict[str, BaseDistribution]:
    """Return the intersection search space of the given trials.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    .. note::
        :class:`~optuna.search_space.IntersectionSearchSpace` provides the same functionality with
        a much faster way. Please consider using it if you want to reduce execution time
        as much as possible.

    Args:
        trials:
            A list of trials.
        include_pruned:
            Whether pruned trials should be included in the search space.

    Returns:
        A dictionary containing the parameter names and parameter's distributions sorted by
        parameter names.
    """

    search_space, _ = _calculate(trials, include_pruned)
    search_space = search_space or {}
    search_space = dict(sorted(search_space.items(), key=lambda x: x[0]))
    return search_space
