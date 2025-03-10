from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution


EPS = 1e-12


class _ParzenEstimatorParameters(NamedTuple):
    consider_prior: bool
    prior_weight: float | None
    consider_magic_clip: bool
    consider_endpoints: bool
    weights: Callable[[int], np.ndarray]
    multivariate: bool
    categorical_distance_func: dict[
        str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
    ]


class _ParzenEstimator:
    def __init__(
        self,
        observations: dict[str, np.ndarray],
        search_space: dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: np.ndarray | None = None,
    ) -> None:
        if parameters.consider_prior:
            if parameters.prior_weight is None:
                raise ValueError("Prior weight must be specified when consider_prior==True.")
            elif parameters.prior_weight <= 0:
                raise ValueError("Prior weight must be positive.")

        self._search_space = search_space

        transformed_observations = self._transform(observations)

        assert predetermined_weights is None or len(transformed_observations) == len(
            predetermined_weights
        )
        weights = (
            predetermined_weights
            if predetermined_weights is not None
            else self._call_weights_func(parameters.weights, len(transformed_observations))
        )

        if len(transformed_observations) == 0:
            weights = np.array([1.0])
        elif parameters.consider_prior:
            assert parameters.prior_weight is not None
            weights = np.append(weights, [parameters.prior_weight])
        weights /= weights.sum()
        self._mixture_distribution = _MixtureOfProductDistribution(
            weights=weights,
            distributions=[
                self._calculate_distributions(
                    transformed_observations[:, i], param, search_space[param], parameters
                )
                for i, param in enumerate(search_space)
            ],
        )

    def sample(self, rng: np.random.RandomState, size: int) -> dict[str, np.ndarray]:
        sampled = self._mixture_distribution.sample(rng, size)
        return self._untransform(sampled)

    def log_pdf(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        transformed_samples = self._transform(samples_dict)
        return self._mixture_distribution.log_pdf(transformed_samples)

    @staticmethod
    def _call_weights_func(weights_func: Callable[[int], np.ndarray], n: int) -> np.ndarray:
        w = np.array(weights_func(n))[:n]
        if np.any(w < 0):
            raise ValueError(
                f"The `weights` function is not allowed to return negative values {w}. "
                + f"The argument of the `weights` function is {n}."
            )
        if len(w) > 0 and np.sum(w) <= 0:
            raise ValueError(
                f"The `weight` function is not allowed to return all-zero values {w}."
                + f" The argument of the `weights` function is {n}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "The `weights`function is not allowed to return infinite or NaN values "
                + f"{w}. The argument of the `weights` function is {n}."
            )

        # TODO(HideakiImamura) Raise `ValueError` if the weight function returns an ndarray of
        # unexpected size.
        return w

    @staticmethod
    def _is_log(dist: BaseDistribution) -> bool:
        return isinstance(dist, (FloatDistribution, IntDistribution)) and dist.log

    def _transform(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(
            [
                (
                    np.log(samples_dict[param])
                    if self._is_log(self._search_space[param])
                    else samples_dict[param]
                )
                for param in self._search_space
            ]
        ).T

    def _untransform(self, samples_array: np.ndarray) -> dict[str, np.ndarray]:
        res = {
            param: (
                np.exp(samples_array[:, i])
                if self._is_log(self._search_space[param])
                else samples_array[:, i]
            )
            for i, param in enumerate(self._search_space)
        }
        # TODO(contramundum53): Remove this line after fixing log-Int hack.
        return {
            param: (
                np.clip(
                    dist.low + np.round((res[param] - dist.low) / dist.step) * dist.step,
                    dist.low,
                    dist.high,
                )
                if isinstance(dist, IntDistribution)
                else res[param]
            )
            for (param, dist) in self._search_space.items()
        }

    def _calculate_distributions(
        self,
        transformed_observations: np.ndarray,
        param_name: str,
        search_space: BaseDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        if isinstance(search_space, CategoricalDistribution):
            return self._calculate_categorical_distributions(
                transformed_observations, param_name, search_space, parameters
            )
        else:
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
            if search_space.log:
                low = np.log(search_space.low)
                high = np.log(search_space.high)
            else:
                low = search_space.low
                high = search_space.high
            step = search_space.step

            # TODO(contramundum53): This is a hack and should be fixed.
            if step is not None and search_space.log:
                low = np.log(search_space.low - step / 2)
                high = np.log(search_space.high + step / 2)
                step = None

            return self._calculate_numerical_distributions(
                transformed_observations, low, high, step, parameters
            )

    def _calculate_categorical_distributions(
        self,
        observations: np.ndarray,
        param_name: str,
        search_space: CategoricalDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        choices = search_space.choices
        n_choices = len(choices)
        if len(observations) == 0:
            return _BatchedCategoricalDistributions(
                weights=np.full((1, n_choices), fill_value=1.0 / n_choices)
            )

        n_kernels = len(observations) + parameters.consider_prior
        assert parameters.prior_weight is not None
        weights = np.full(
            shape=(n_kernels, n_choices),
            fill_value=parameters.prior_weight / n_kernels,
        )
        observed_indices = observations.astype(int)
        if param_name in parameters.categorical_distance_func:
            # TODO(nabenabe0928): Think about how to handle combinatorial explosion.
            # The time complexity is O(n_choices * used_indices.size), so n_choices cannot be huge.
            used_indices, rev_indices = np.unique(observed_indices, return_inverse=True)
            dist_func = parameters.categorical_distance_func[param_name]
            dists = np.array([[dist_func(choices[i], c) for c in choices] for i in used_indices])
            coef = np.log(n_kernels / parameters.prior_weight) * np.log(n_choices) / np.log(6)
            cat_weights = np.exp(-((dists / np.max(dists, axis=1)[:, np.newaxis]) ** 2) * coef)
            weights[: len(observed_indices)] = cat_weights[rev_indices]
        else:
            weights[np.arange(len(observed_indices)), observed_indices] += 1

        weights /= weights.sum(axis=1, keepdims=True)
        return _BatchedCategoricalDistributions(weights)

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        low: float,
        high: float,
        step: float | None,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        step_or_0 = step or 0

        mus = observations
        consider_prior = parameters.consider_prior or len(observations) == 0

        def compute_sigmas() -> np.ndarray:
            if parameters.multivariate:
                SIGMA0_MAGNITUDE = 0.2
                sigma = (
                    SIGMA0_MAGNITUDE
                    * max(len(observations), 1) ** (-1.0 / (len(self._search_space) + 4))
                    * (high - low + step_or_0)
                )
                sigmas = np.full(shape=(len(observations),), fill_value=sigma)
            else:
                # TODO(contramundum53): Remove dependency on prior_mu
                prior_mu = 0.5 * (low + high)
                mus_with_prior = np.append(mus, prior_mu) if consider_prior else mus

                sorted_indices = np.argsort(mus_with_prior)
                sorted_mus = mus_with_prior[sorted_indices]
                sorted_mus_with_endpoints = np.empty(len(mus_with_prior) + 2, dtype=float)
                sorted_mus_with_endpoints[0] = low - step_or_0 / 2
                sorted_mus_with_endpoints[1:-1] = sorted_mus
                sorted_mus_with_endpoints[-1] = high + step_or_0 / 2

                sorted_sigmas = np.maximum(
                    sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                    sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
                )

                if not parameters.consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                    sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                    sorted_sigmas[-1] = (
                        sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]
                    )

                sigmas = sorted_sigmas[np.argsort(sorted_indices)][: len(observations)]

            # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
            maxsigma = 1.0 * (high - low + step_or_0)
            if parameters.consider_magic_clip:
                # TODO(contramundum53): Remove dependency of minsigma on consider_prior.
                minsigma = (
                    1.0
                    * (high - low + step_or_0)
                    / min(100.0, (1.0 + len(observations) + consider_prior))
                )
            else:
                minsigma = EPS
            return np.asarray(np.clip(sigmas, minsigma, maxsigma))

        sigmas = compute_sigmas()

        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low + step_or_0)
            mus = np.append(mus, [prior_mu])
            sigmas = np.append(sigmas, [prior_sigma])

        if step is None:
            return _BatchedTruncNormDistributions(mus, sigmas, low, high)
        else:
            return _BatchedDiscreteTruncNormDistributions(mus, sigmas, low, high, step)
