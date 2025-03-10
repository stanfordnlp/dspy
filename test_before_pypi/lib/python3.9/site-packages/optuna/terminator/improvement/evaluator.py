from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import intersection_search_space
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:

    from optuna._gp import acqf
    from optuna._gp import gp
    from optuna._gp import optim_sample
    from optuna._gp import prior
    from optuna._gp import search_space as gp_search_space
else:
    from optuna._imports import _LazyImport

    gp = _LazyImport("optuna._gp.gp")
    optim_sample = _LazyImport("optuna._gp.optim_sample")
    acqf = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")
    gp_search_space = _LazyImport("optuna._gp.search_space")

DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


def _get_beta(n_params: int, n_trials: int, delta: float = 0.1) -> float:
    # TODO(nabenabe0928): Check the original implementation to verify.
    # Especially, |D| seems to be the domain size, but not the dimension based on Theorem 1.
    beta = 2 * np.log(n_params * n_trials**2 * np.pi**2 / 6 / delta)

    # The following div is according to the original paper: "We then further scale it down
    # by a factor of 5 as defined in the experiments in
    # `Srinivas et al. (2010) <https://dl.acm.org/doi/10.5555/3104322.3104451>`__"
    beta /= 5

    return beta


def _compute_standardized_regret_bound(
    kernel_params: gp.KernelParamsTensor,
    search_space: gp_search_space.SearchSpace,
    normalized_top_n_params: np.ndarray,
    standarized_top_n_values: np.ndarray,
    delta: float = 0.1,
    optimize_n_samples: int = 2048,
    rng: np.random.RandomState | None = None,
) -> float:
    """
    # In the original paper, f(x) was intended to be minimized, but here we would like to
    # maximize f(x). Hence, the following changes happen:
    #     1. min(ucb) over top trials becomes max(lcb) over top trials, and
    #     2. min(lcb) over the search space becomes max(ucb) over the search space, and
    #     3. Regret bound becomes max(ucb) over the search space minus max(lcb) over top trials.
    """

    n_trials, n_params = normalized_top_n_params.shape

    # calculate max_ucb
    beta = _get_beta(n_params, n_trials, delta)
    ucb_acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.UCB,
        kernel_params=kernel_params,
        search_space=search_space,
        X=normalized_top_n_params,
        Y=standarized_top_n_values,
        beta=beta,
    )
    # UCB over the search space. (Original: LCB over the search space. See Change 1 above.)
    standardized_ucb_value = max(
        acqf.eval_acqf_no_grad(ucb_acqf_params, normalized_top_n_params).max(),
        optim_sample.optimize_acqf_sample(ucb_acqf_params, n_samples=optimize_n_samples, rng=rng)[
            1
        ],
    )

    # calculate min_lcb
    lcb_acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.LCB,
        kernel_params=kernel_params,
        search_space=search_space,
        X=normalized_top_n_params,
        Y=standarized_top_n_values,
        beta=beta,
    )
    # LCB over the top trials. (Original: UCB over the top trials. See Change 2 above.)
    standardized_lcb_value = np.max(
        acqf.eval_acqf_no_grad(lcb_acqf_params, normalized_top_n_params)
    )

    # max(UCB) - max(LCB). (Original: min(UCB) - min(LCB). See Change 3 above.)
    return standardized_ucb_value - standardized_lcb_value  # standardized regret bound


@experimental_class("3.2.0")
class BaseImprovementEvaluator(metaclass=abc.ABCMeta):
    """Base class for improvement evaluators."""

    @abc.abstractmethod
    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        pass


@experimental_class("3.2.0")
class RegretBoundEvaluator(BaseImprovementEvaluator):
    """An error evaluator for upper bound on the regret with high-probability confidence.

    This evaluator evaluates the regret of current best solution, which defined as the difference
    between the objective value of the best solution and of the global optimum. To be specific,
    this evaluator calculates the upper bound on the regret based on the fact that empirical
    estimator of the objective function is bounded by lower and upper confidence bounds with
    high probability under the Gaussian process model assumption.

    Args:
        top_trials_ratio:
            A ratio of top trials to be considered when estimating the regret. Default to 0.5.
        min_n_trials:
            A minimum number of complete trials to estimate the regret. Default to 20.
        seed:
            Seed for random number generator.

    For further information about this evaluator, please refer to the following paper:

    - `Automatic Termination for Hyperparameter Optimization <https://proceedings.mlr.press/v188/makarova22a.html>`__
    """  # NOQA: E501

    def __init__(
        self,
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        seed: int | None = None,
    ) -> None:
        self._top_trials_ratio = top_trials_ratio
        self._min_n_trials = min_n_trials
        self._log_prior = prior.default_log_prior
        self._minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
        self._optimize_n_samples = 2048
        self._rng = LazyRandomState(seed)

    def _get_top_n(
        self, normalized_params: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert len(normalized_params) == len(values)
        n_trials = len(normalized_params)
        top_n = np.clip(int(n_trials * self._top_trials_ratio), self._min_n_trials, n_trials)
        top_n_val = np.partition(values, n_trials - top_n)[n_trials - top_n]
        top_n_mask = values >= top_n_val
        return normalized_params[top_n_mask], values[top_n_mask]

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        optuna_search_space = intersection_search_space(trials)
        self._validate_input(trials, optuna_search_space)

        complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]

        # _gp module assumes that optimization direction is maximization
        sign = -1 if study_direction == StudyDirection.MINIMIZE else 1
        values = np.array([t.value for t in complete_trials]) * sign
        search_space, normalized_params = gp_search_space.get_search_space_and_normalized_params(
            complete_trials, optuna_search_space
        )
        normalized_top_n_params, top_n_values = self._get_top_n(normalized_params, values)
        top_n_values_mean = top_n_values.mean()
        top_n_values_std = max(1e-10, top_n_values.std())
        standarized_top_n_values = (top_n_values - top_n_values_mean) / top_n_values_std

        kernel_params = gp.fit_kernel_params(
            X=normalized_top_n_params,
            Y=standarized_top_n_values,
            is_categorical=(search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL),
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            # TODO(contramundum53): Add option to specify this.
            deterministic_objective=False,
            # TODO(y0z): Add `kernel_params_cache` to speedup.
            initial_kernel_params=None,
        )

        standardized_regret_bound = _compute_standardized_regret_bound(
            kernel_params,
            search_space,
            normalized_top_n_params,
            standarized_top_n_values,
            rng=self._rng.rng,
        )
        return standardized_regret_bound * top_n_values_std  # regret bound

    @classmethod
    def _validate_input(
        cls, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> None:
        if len([t for t in trials if t.state == TrialState.COMPLETE]) == 0:
            raise ValueError(
                "Because no trial has been completed yet, the regret bound cannot be evaluated."
            )

        if len(search_space) == 0:
            raise ValueError(
                "The intersection search space is empty. This condition is not supported by "
                f"{cls.__name__}."
            )


@experimental_class("3.4.0")
class BestValueStagnationEvaluator(BaseImprovementEvaluator):
    """Evaluates the stagnation period of the best value in an optimization process.

    This class is initialized with a maximum stagnation period (`max_stagnation_trials`)
    and is designed to evaluate the remaining trials before reaching this maximum period
    of allowed stagnation. If this remaining trials reach zero, the trial terminates.
    Therefore, the default error evaluator is instantiated by StaticErrorEvaluator(const=0).

    Args:
        max_stagnation_trials:
            The maximum number of trials allowed for stagnation.
    """

    def __init__(self, max_stagnation_trials: int = 30) -> None:
        if max_stagnation_trials < 0:
            raise ValueError("The maximum number of stagnant trials must not be negative.")
        self._max_stagnation_trials = max_stagnation_trials

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        self._validate_input(trials)
        is_maximize_direction = True if (study_direction == StudyDirection.MAXIMIZE) else False
        trials = [t for t in trials if t.state == TrialState.COMPLETE]
        current_step = len(trials) - 1

        best_step = 0
        for i, trial in enumerate(trials):
            best_value = trials[best_step].value
            current_value = trial.value
            assert best_value is not None
            assert current_value is not None
            if is_maximize_direction and (best_value < current_value):
                best_step = i
            elif (not is_maximize_direction) and (best_value > current_value):
                best_step = i

        return self._max_stagnation_trials - (current_step - best_step)

    @classmethod
    def _validate_input(cls, trials: list[FrozenTrial]) -> None:
        if len([t for t in trials if t.state == TrialState.COMPLETE]) == 0:
            raise ValueError(
                "Because no trial has been completed yet, the improvement cannot be evaluated."
            )
