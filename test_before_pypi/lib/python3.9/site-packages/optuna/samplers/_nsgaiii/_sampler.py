from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import hashlib
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    NSGAIIIElitePopulationSelectionStrategy,
)
from optuna.samplers._random import RandomSampler
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "nsga3:generation"
_POPULATION_CACHE_KEY_PREFIX = "nsga3:population"


@experimental_class("3.2.0")
class NSGAIIISampler(BaseSampler):
    """Multi-objective sampler using the NSGA-III algorithm.

    NSGA-III stands for "Nondominated Sorting Genetic Algorithm III",
    which is a modified version of NSGA-II for many objective optimization problem.

    For further information about NSGA-III, please refer to the following papers:

    - `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
      Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints
      <https://doi.org/10.1109/TEVC.2013.2281535>`__
    - `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
      Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive
      Approach
      <https://doi.org/10.1109/TEVC.2013.2281534>`__

    Args:
        reference_points:
            A 2 dimension ``numpy.ndarray`` with objective dimension columns. Represents
            a list of reference points which is used to determine who to survive.
            After non-dominated sort, who out of borderline front are going to survived is
            determined according to how sparse the closest reference point of each individual is.
            In the default setting the algorithm uses `uniformly` spread points to diversify the
            result. It is also possible to reflect your `preferences` by giving an arbitrary set of
            `target` points since the algorithm prioritizes individuals around reference points.

        dividing_parameter:
            A parameter to determine the density of default reference points. This parameter
            determines how many divisions are made between reference points on each axis. The
            smaller this value is, the less reference points you have. The default value is 3.
            Note that this parameter is not used when ``reference_points`` is not :obj:`None`.

    .. note::
        Other parameters than ``reference_points`` and ``dividing_parameter`` are the same as
        :class:`~optuna.samplers.NSGAIISampler`.

    """

    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: int | None = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        reference_points: np.ndarray | None = None,
        dividing_parameter: int = 3,
        elite_population_selection_strategy: (
            Callable[[Study, list[FrozenTrial]], list[FrozenTrial]] | None
        ) = None,
        child_generation_strategy: (
            Callable[[Study, dict[str, BaseDistribution], list[FrozenTrial]], dict[str, Any]]
            | None
        ) = None,
        after_trial_strategy: (
            Callable[[Study, FrozenTrial, TrialState, Sequence[float] | None], None] | None
        ) = None,
    ) -> None:
        # TODO(ohta): Reconsider the default value of each parameter.

        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        if crossover is None:
            crossover = UniformCrossover(swapping_prob)

        if not isinstance(crossover, BaseCrossover):
            raise ValueError(
                f"'{crossover}' is not a valid crossover."
                " For valid crossovers see"
                " https://optuna.readthedocs.io/en/stable/reference/samplers.html."
            )

        if population_size < crossover.n_parents:
            raise ValueError(
                f"Using {crossover},"
                f" the population size should be greater than or equal to {crossover.n_parents}."
                f" The specified `population_size` is {population_size}."
            )

        self._population_size = population_size
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = LazyRandomState(seed)
        self._constraints_func = constraints_func
        self._search_space = IntersectionSearchSpace()

        self._elite_population_selection_strategy = (
            elite_population_selection_strategy
            or NSGAIIIElitePopulationSelectionStrategy(
                population_size=population_size,
                constraints_func=constraints_func,
                reference_points=reference_points,
                dividing_parameter=dividing_parameter,
                rng=self._rng,
            )
        )
        self._child_generation_strategy = (
            child_generation_strategy
            or NSGAIIChildGenerationStrategy(
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                swapping_prob=swapping_prob,
                crossover=crossover,
                constraints_func=constraints_func,
                rng=self._rng,
            )
        )
        self._after_trial_strategy = after_trial_strategy or NSGAIIAfterTrialStrategy(
            constraints_func=constraints_func
        )

    def reseed_rng(self) -> None:
        self._random_sampler.reseed_rng()
        self._rng.rng.seed()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # The `untransform` method of `optuna._transform._SearchSpaceTransform`
                # does not assume a single value,
                # so single value objects are not sampled with the `sample_relative` method,
                # but with the `sample_independent` method.
                continue
            search_space[name] = distribution
        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        parent_generation, parent_population = self._collect_parent_population(study)

        generation = parent_generation + 1
        study._storage.set_trial_system_attr(trial._trial_id, _GENERATION_KEY, generation)

        if parent_generation < 0:
            return {}

        return self._child_generation_strategy(study, search_space, parent_population)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # Following parameters are randomly sampled here.
        # 1. A parameter in the initial population/first generation.
        # 2. A parameter to mutate.
        # 3. A parameter excluded from the intersection search space.

        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _collect_parent_population(self, study: Study) -> tuple[int, list[FrozenTrial]]:
        trials = study.get_trials(deepcopy=False)

        generation_to_runnings = defaultdict(list)
        generation_to_population = defaultdict(list)
        for trial in trials:
            if _GENERATION_KEY not in trial.system_attrs:
                continue

            generation = trial.system_attrs[_GENERATION_KEY]
            if trial.state != optuna.trial.TrialState.COMPLETE:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    generation_to_runnings[generation].append(trial)
                continue

            # Do not use trials whose states are not COMPLETE, or `constraint` will be unavailable.
            generation_to_population[generation].append(trial)

        hasher = hashlib.sha256()
        parent_population: list[FrozenTrial] = []
        parent_generation = -1
        while True:
            generation = parent_generation + 1
            population = generation_to_population[generation]

            # Under multi-worker settings, the population size might become larger than
            # `self._population_size`.
            if len(population) < self._population_size:
                break

            # [NOTE]
            # It's generally safe to assume that once the above condition is satisfied,
            # there are no additional individuals added to the generation (i.e., the members of
            # the generation have been fixed).
            # If the number of parallel workers is huge, this assumption can be broken, but
            # this is a very rare case and doesn't significantly impact optimization performance.
            # So we can ignore the case.

            # The cache key is calculated based on the key of the previous generation and
            # the remaining running trials in the current population.
            # If there are no running trials, the new cache key becomes exactly the same as
            # the previous one, and the cached content will be overwritten. This allows us to
            # skip redundant cache key calculations when this method is called for the subsequent
            # trials.
            for trial in generation_to_runnings[generation]:
                hasher.update(bytes(str(trial.number), "utf-8"))

            cache_key = "{}:{}".format(_POPULATION_CACHE_KEY_PREFIX, hasher.hexdigest())
            study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
            cached_generation, cached_population_numbers = study_system_attrs.get(
                cache_key, (-1, [])
            )
            if cached_generation >= generation:
                generation = cached_generation
                population = [trials[n] for n in cached_population_numbers]
            else:
                population.extend(parent_population)
                population = self._elite_population_selection_strategy(study, population)

                # To reduce the number of system attribute entries,
                # we cache the population information only if there are no running trials
                # (i.e., the information of the population has been fixed).
                # Usually, if there are no too delayed running trials, the single entry
                # will be used.
                if len(generation_to_runnings[generation]) == 0:
                    population_numbers = [t.number for t in population]
                    study._storage.set_study_system_attr(
                        study._study_id, cache_key, (generation, population_numbers)
                    )

            parent_generation = generation
            parent_population = population

        return parent_generation, parent_population

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._after_trial_strategy(study, trial, state, values)
        self._random_sampler.after_trial(study, trial, state, values)
