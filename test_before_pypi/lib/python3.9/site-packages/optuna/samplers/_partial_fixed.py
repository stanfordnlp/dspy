from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING
import warnings

from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("2.4.0")
class PartialFixedSampler(BaseSampler):
    """Sampler with partially fixed parameters.

    Example:

        After several steps of optimization, you can fix the value of ``y`` and re-optimize it.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x**2 + y


            study = optuna.create_study()
            study.optimize(objective, n_trials=10)

            best_params = study.best_params
            fixed_params = {"y": best_params["y"]}
            partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)

            study.sampler = partial_sampler
            study.optimize(objective, n_trials=10)

    Args:

        fixed_params:
            A dictionary of parameters to be fixed.

        base_sampler:
            A sampler which samples unfixed parameters.

    """

    def __init__(self, fixed_params: dict[str, Any], base_sampler: BaseSampler) -> None:
        self._fixed_params = fixed_params
        self._base_sampler = base_sampler

    def reseed_rng(self) -> None:
        self._base_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space = self._base_sampler.infer_relative_search_space(study, trial)

        # Remove fixed params from relative search space to return fixed values.
        for param_name in self._fixed_params.keys():
            if param_name in search_space:
                del search_space[param_name]

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        # Fixed params are never sampled here.
        return self._base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if param_name not in self._fixed_params:
            # Unfixed params are sampled here.
            return self._base_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        else:
            # Fixed params are sampled here.
            # Check if a parameter value is contained in the range of this distribution.
            param_value = self._fixed_params[param_name]

            param_value_in_internal_repr = param_distribution.to_internal_repr(param_value)
            contained = param_distribution._contains(param_value_in_internal_repr)

            if not contained:
                warnings.warn(
                    f"Fixed parameter '{param_name}' with value {param_value} is out of range "
                    f"for distribution {param_distribution}."
                )
            return param_value

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._base_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._base_sampler.after_trial(study, trial, state, values)
