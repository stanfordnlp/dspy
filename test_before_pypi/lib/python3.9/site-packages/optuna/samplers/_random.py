from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna import distributions
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    from optuna.study import Study


class RandomSampler(BaseSampler):
    """Sampler using random sampling.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import RandomSampler


            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                return x**2


            study = optuna.create_study(sampler=RandomSampler())
            study.optimize(objective, n_trials=10)

    Args:
        seed: Seed for random number generator.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = LazyRandomState(seed)

    def reseed_rng(self) -> None:
        self._rng.rng.seed()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        search_space = {param_name: param_distribution}
        trans = _SearchSpaceTransform(search_space)
        trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])

        return trans.untransform(trans_params)[param_name]
