from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    from optuna.study import Study


_NUMERICAL_DISTRIBUTIONS = (
    FloatDistribution,
    IntDistribution,
)


def _try_crossover(
    parents: list[FrozenTrial],
    crossover: BaseCrossover,
    study: Study,
    rng: np.random.RandomState,
    swapping_prob: float,
    categorical_search_space: dict[str, BaseDistribution],
    numerical_search_space: dict[str, BaseDistribution],
    numerical_transform: _SearchSpaceTransform | None,
) -> dict[str, Any]:
    child_params: dict[str, Any] = {}

    if len(categorical_search_space) > 0:
        parents_categorical_params = np.array(
            [
                [parent.params[p] for p in categorical_search_space]
                for parent in [parents[0], parents[-1]]
            ],
            dtype=object,
        )

        child_categorical_array = _inlined_categorical_uniform_crossover(
            parents_categorical_params, rng, swapping_prob, categorical_search_space
        )
        child_categorical_params = {
            param: value for param, value in zip(categorical_search_space, child_categorical_array)
        }
        child_params.update(child_categorical_params)

    if numerical_transform is None:
        return child_params

    # The following is applied only for numerical parameters.
    parents_numerical_params = np.stack(
        [
            numerical_transform.transform(
                {
                    param_key: parent.params[param_key]
                    for param_key in numerical_search_space.keys()
                }
            )
            for parent in parents
        ]
    )  # Parent individual with NUMERICAL_DISTRIBUTIONS parameter.

    child_numerical_array = crossover.crossover(
        parents_numerical_params, rng, study, numerical_transform.bounds
    )
    child_numerical_params = numerical_transform.untransform(child_numerical_array)
    child_params.update(child_numerical_params)

    return child_params


def perform_crossover(
    crossover: BaseCrossover,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    search_space: dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> dict[str, Any]:
    numerical_search_space: dict[str, BaseDistribution] = {}
    categorical_search_space: dict[str, BaseDistribution] = {}
    for key, value in search_space.items():
        if isinstance(value, _NUMERICAL_DISTRIBUTIONS):
            numerical_search_space[key] = value
        else:
            categorical_search_space[key] = value

    numerical_transform: _SearchSpaceTransform | None = None
    if len(numerical_search_space) != 0:
        numerical_transform = _SearchSpaceTransform(numerical_search_space)

    while True:  # Repeat while parameters lie outside search space boundaries.
        parents = _select_parents(crossover, study, parent_population, rng, dominates)
        child_params = _try_crossover(
            parents,
            crossover,
            study,
            rng,
            swapping_prob,
            categorical_search_space,
            numerical_search_space,
            numerical_transform,
        )

        if _is_contained(child_params, search_space):
            break

    return child_params


def _select_parents(
    crossover: BaseCrossover,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> list[FrozenTrial]:
    parents: list[FrozenTrial] = []
    for _ in range(crossover.n_parents):
        parent = _select_parent(
            study, [t for t in parent_population if t not in parents], rng, dominates
        )
        parents.append(parent)

    return parents


def _select_parent(
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> FrozenTrial:
    population_size = len(parent_population)
    candidate0 = parent_population[rng.choice(population_size)]
    candidate1 = parent_population[rng.choice(population_size)]

    # TODO(ohta): Consider crowding distance.
    if dominates(candidate0, candidate1, study.directions):
        return candidate0
    else:
        return candidate1


def _is_contained(params: dict[str, Any], search_space: dict[str, BaseDistribution]) -> bool:
    for param_name in params.keys():
        param, param_distribution = params[param_name], search_space[param_name]

        if not param_distribution._contains(param_distribution.to_internal_repr(param)):
            return False
    return True


def _inlined_categorical_uniform_crossover(
    parent_params: np.ndarray,
    rng: np.random.RandomState,
    swapping_prob: float,
    search_space: dict[str, BaseDistribution],
) -> np.ndarray:
    # We can't use uniform crossover implementation of `BaseCrossover` for
    # parameters from `CategoricalDistribution`, since categorical params are
    # passed to crossover untransformed, which is not what `BaseCrossover`
    # implementations expect.
    n_categorical_params = len(search_space)
    masks = (rng.rand(n_categorical_params) >= swapping_prob).astype(int)
    return parent_params[masks, range(n_categorical_params)]
