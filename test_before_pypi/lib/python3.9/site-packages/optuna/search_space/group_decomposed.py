from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


class _SearchSpaceGroup:
    def __init__(self) -> None:
        self._search_spaces: list[dict[str, BaseDistribution]] = []

    @property
    def search_spaces(self) -> list[dict[str, BaseDistribution]]:
        return self._search_spaces

    def add_distributions(self, distributions: dict[str, BaseDistribution]) -> None:
        dist_keys = set(distributions.keys())
        next_search_spaces = []

        for search_space in self._search_spaces:
            keys = set(search_space.keys())

            next_search_spaces.append({name: search_space[name] for name in keys & dist_keys})
            next_search_spaces.append({name: search_space[name] for name in keys - dist_keys})

            dist_keys -= keys

        next_search_spaces.append({name: distributions[name] for name in dist_keys})
        self._search_spaces = list(
            filter(lambda search_space: len(search_space) > 0, next_search_spaces)
        )


class _GroupDecomposedSearchSpace:
    def __init__(self, include_pruned: bool = False) -> None:
        self._search_space = _SearchSpaceGroup()
        self._study_id: int | None = None
        self._include_pruned = include_pruned

    def calculate(self, study: Study) -> _SearchSpaceGroup:
        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when
            # :class:`~optuna.storages.InMemoryStorage` is used because
            # :func:`~optuna.storages.InMemoryStorage.create_new_study`
            # always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`_GroupDecomposedSearchSpace` cannot handle multiple studies.")

        states_of_interest: tuple[TrialState, ...]
        if self._include_pruned:
            states_of_interest = (TrialState.COMPLETE, TrialState.PRUNED)
        else:
            states_of_interest = (TrialState.COMPLETE,)

        for trial in study._get_trials(deepcopy=False, states=states_of_interest, use_cache=False):
            self._search_space.add_distributions(trial.distributions)

        return copy.deepcopy(self._search_space)
