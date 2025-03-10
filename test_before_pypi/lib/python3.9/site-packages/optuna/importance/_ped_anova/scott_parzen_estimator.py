from __future__ import annotations

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.trial import FrozenTrial


class _ScottParzenEstimator(_ParzenEstimator):
    """1D ParzenEstimator using the bandwidth selection by Scott's rule."""

    def __init__(
        self,
        param_name: str,
        dist: IntDistribution | CategoricalDistribution,
        counts: np.ndarray,
        consider_prior: bool,
        prior_weight: float,
    ):
        assert isinstance(dist, (CategoricalDistribution, IntDistribution))
        assert not isinstance(dist, IntDistribution) or dist.low == 0
        n_choices = dist.high + 1 if isinstance(dist, IntDistribution) else len(dist.choices)
        assert len(counts) == n_choices, counts

        self._n_steps = len(counts)
        self._param_name = param_name
        self._counts = counts.copy()
        super().__init__(
            observations={param_name: np.arange(self._n_steps)[counts > 0.0]},
            search_space={param_name: dist},
            parameters=_ParzenEstimatorParameters(
                consider_prior=consider_prior,
                prior_weight=prior_weight,
                consider_magic_clip=False,
                consider_endpoints=False,
                weights=lambda x: np.empty(0),
                multivariate=True,
                categorical_distance_func={},
            ),
            predetermined_weights=counts[counts > 0.0],
        )

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        low: float,  # The type is actually int, but typing follows the original.
        high: float,  # The type is actually int, but typing follows the original.
        step: float | None,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        # NOTE: The Optuna TPE bandwidth selection is too wide for this analysis.
        # So use the Scott's rule by Scott, D.W. (1992),
        # Multivariate Density Estimation: Theory, Practice, and Visualization.
        assert step is not None and np.isclose(step, 1.0), "MyPy redefinition."

        n_trials = np.sum(self._counts)
        counts_non_zero = self._counts[self._counts > 0]
        weights = counts_non_zero / n_trials
        mus = np.arange(self.n_steps)[self._counts > 0]
        mean_est = mus @ weights
        sigma_est = np.sqrt((mus - mean_est) ** 2 @ counts_non_zero / max(1, n_trials - 1))

        count_cum = np.cumsum(counts_non_zero)
        idx_q25 = np.searchsorted(count_cum, n_trials // 4, side="left")
        idx_q75 = np.searchsorted(count_cum, n_trials * 3 // 4, side="right")
        interquantile_range = mus[min(mus.size - 1, idx_q75)] - mus[idx_q25]
        sigma_est = 1.059 * min(interquantile_range / 1.34, sigma_est) * n_trials ** (-0.2)
        # To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        sigma_min = 0.5 / 1.64
        sigmas = np.full_like(mus, max(sigma_est, sigma_min), dtype=np.float64)
        if parameters.consider_prior:
            mus = np.append(mus, [0.5 * (low + high)])
            sigmas = np.append(sigmas, [1.0 * (high - low + 1)])

        return _BatchedDiscreteTruncNormDistributions(
            mu=mus, sigma=sigmas, low=0, high=self.n_steps - 1, step=1
        )

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf({self._param_name: samples}))


def _get_grids_and_grid_indices_of_trials(
    param_name: str,
    dist: IntDistribution | FloatDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
) -> tuple[int, np.ndarray]:
    assert isinstance(dist, (FloatDistribution, IntDistribution)), "Unexpected distribution."
    if isinstance(dist, IntDistribution) and dist.log:
        log2_domain_size = int(np.ceil(np.log(dist.high - dist.low + 1) / np.log(2))) + 1
        n_steps = min(log2_domain_size, n_steps)
    elif dist.step is not None:
        assert not dist.log, "log must be False when step is not None."
        n_steps = min(round((dist.high - dist.low) / dist.step) + 1, n_steps)

    scaler = np.log if dist.log else np.asarray
    grids = np.linspace(scaler(dist.low), scaler(dist.high), n_steps)  # type: ignore[operator]
    params = scaler([t.params[param_name] for t in trials])  # type: ignore[operator]
    step_size = grids[1] - grids[0]
    # grids[indices[n] - 1] < param - step_size / 2 <= grids[indices[n]]
    indices = np.searchsorted(grids, params - step_size / 2)
    return grids.size, indices


def _count_numerical_param_in_grid(
    param_name: str,
    dist: IntDistribution | FloatDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
) -> np.ndarray:
    n_grids, grid_indices_of_trials = _get_grids_and_grid_indices_of_trials(
        param_name, dist, trials, n_steps
    )
    unique_vals, counts_in_unique = np.unique(grid_indices_of_trials, return_counts=True)
    counts = np.zeros(n_grids, dtype=np.int32)
    counts[unique_vals] += counts_in_unique
    return counts


def _count_categorical_param_in_grid(
    param_name: str, dist: CategoricalDistribution, trials: list[FrozenTrial]
) -> np.ndarray:
    cat_indices = [int(dist.to_internal_repr(t.params[param_name])) for t in trials]
    unique_vals, counts_in_unique = np.unique(cat_indices, return_counts=True)
    counts = np.zeros(len(dist.choices), dtype=np.int32)
    counts[unique_vals] += counts_in_unique
    return counts


def _build_parzen_estimator(
    param_name: str,
    dist: BaseDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
    consider_prior: bool,
    prior_weight: float,
) -> _ScottParzenEstimator:
    rounded_dist: IntDistribution | CategoricalDistribution
    if isinstance(dist, (IntDistribution, FloatDistribution)):
        counts = _count_numerical_param_in_grid(param_name, dist, trials, n_steps)
        rounded_dist = IntDistribution(low=0, high=counts.size - 1)
    elif isinstance(dist, CategoricalDistribution):
        counts = _count_categorical_param_in_grid(param_name, dist, trials)
        rounded_dist = dist
    else:
        assert False, f"Got an unknown dist with the type {type(dist)}."

    # counts.astype(float) is necessary for weight calculation in ParzenEstimator.
    return _ScottParzenEstimator(
        param_name, rounded_dist, counts.astype(np.float64), consider_prior, prior_weight
    )
