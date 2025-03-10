from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import hashlib
from typing import Any
from typing import TYPE_CHECKING

import optuna
from optuna._experimental import warn_experimental_argument
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.samplers.nsgaii._elite_population_selection_strategy import (
    NSGAIIElitePopulationSelectionStrategy,
)
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "nsga2:generation"
_POPULATION_CACHE_KEY_PREFIX = "nsga2:population"


class NSGAIISampler(BaseSampler):
    """Multi-objective sampler using the NSGA-II algorithm.

    NSGA-II stands for "Nondominated Sorting Genetic Algorithm II",
    which is a well known, fast and elitist multi-objective genetic algorithm.

    For further information about NSGA-II, please refer to the following paper:

    - `A fast and elitist multiobjective genetic algorithm: NSGA-II
      <https://doi.org/10.1109/4235.996017>`__

    .. note::
        :class:`~optuna.samplers.TPESampler` became much faster in v4.0.0 and supports several
        features not supported by ``NSGAIISampler`` such as handling of dynamic search
        space and categorical distance. To use :class:`~optuna.samplers.TPESampler`, you need to
        explicitly specify the sampler as follows:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                f1 = x**2 + y
                f2 = -((x - 2) ** 2 + y)
                return f1, f2


            # We minimize the first objective and maximize the second objective.
            sampler = optuna.samplers.TPESampler()
            study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
            study.optimize(objective, n_trials=100)

        Please also check `our article
        <https://medium.com/optuna/significant-speed-up-of-multi-objective-tpesampler-in-optuna-v4-0-0-2bacdcd1d99b>`__
        for more details of the speedup in v4.0.0.

    Args:
        population_size:
            Number of individuals (trials) in a generation.
            ``population_size`` must be greater than or equal to ``crossover.n_parents``.
            For :class:`~optuna.samplers.nsgaii.UNDXCrossover` and
            :class:`~optuna.samplers.nsgaii.SPXCrossover`, ``n_parents=3``, and for the other
            algorithms, ``n_parents=2``.

        mutation_prob:
            Probability of mutating each parameter when creating a new individual.
            If :obj:`None` is specified, the value ``1.0 / len(parent_trial.params)`` is used
            where ``parent_trial`` is the parent trial of the target individual.

        crossover:
            Crossover to be applied when creating child individuals.
            The available crossovers are listed here:
            https://optuna.readthedocs.io/en/stable/reference/samplers/nsgaii.html.

            :class:`~optuna.samplers.nsgaii.UniformCrossover` is always applied to parameters
            sampled from :class:`~optuna.distributions.CategoricalDistribution`, and by
            default for parameters sampled from other distributions unless this argument
            is specified.

            For more information on each of the crossover method, please refer to
            specific crossover documentation.

        crossover_prob:
            Probability that a crossover (parameters swapping between parents) will occur
            when creating a new individual.

        swapping_prob:
            Probability of swapping each parameter of the parents during crossover.

        seed:
            Seed for random number generator.

        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.

            The constraints are handled by the constrained domination. A trial x is said to
            constrained-dominate a trial y, if any of the following conditions is true:

            1. Trial x is feasible and trial y is not.
            2. Trial x and y are both infeasible, but trial x has a smaller overall violation.
            3. Trial x and y are feasible and trial x dominates trial y.

            .. note::
                Added in v2.5.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.5.0.

        elite_population_selection_strategy:
            The selection strategy for determining the individuals to survive from the current
            population pool. Default to :obj:`None`.

            .. note::
                The arguments ``elite_population_selection_strategy`` was added in v3.3.0 as an
                experimental feature. The interface may change in newer versions without prior
                notice.
                See https://github.com/optuna/optuna/releases/tag/v3.3.0.

        child_generation_strategy:
            The strategy for generating child parameters from parent trials. Defaults to
            :obj:`None`.

            .. note::
                The arguments ``child_generation_strategy`` was added in v3.3.0 as an experimental
                feature. The interface may change in newer versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.3.0.

        after_trial_strategy:
            A set of procedure to be conducted after each trial. Defaults to :obj:`None`.

            .. note::
                The arguments ``after_trial_strategy`` was added in v3.3.0 as an experimental
                feature. The interface may change in newer versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.3.0.
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

        if constraints_func is not None:
            warn_experimental_argument("constraints_func")
        if after_trial_strategy is not None:
            warn_experimental_argument("after_trial_strategy")

        if child_generation_strategy is not None:
            warn_experimental_argument("child_generation_strategy")

        if elite_population_selection_strategy is not None:
            warn_experimental_argument("elite_population_selection_strategy")

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
            or NSGAIIElitePopulationSelectionStrategy(
                population_size=population_size, constraints_func=constraints_func
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
        trials = study._get_trials(deepcopy=False, use_cache=True)

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
