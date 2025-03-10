from __future__ import annotations

from collections.abc import Sequence
import threading
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna import logging
from optuna._experimental import experimental_class
from optuna._imports import _LazyImport
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


_logger = logging.get_logger(__name__)

_SUGGESTED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)
_threading_lock = threading.Lock()


@experimental_class("3.0.0")
class QMCSampler(BaseSampler):
    """A Quasi Monte Carlo Sampler that generates low-discrepancy sequences.

    Quasi Monte Carlo (QMC) sequences are designed to have lower discrepancies than
    standard random sequences. They are known to perform better than the standard
    random sequences in hyperparameter optimization.

    For further information about the use of QMC sequences for hyperparameter optimization,
    please refer to the following paper:

    - `Bergstra, James, and Yoshua Bengio. Random search for hyper-parameter optimization.
      Journal of machine learning research 13.2, 2012.
      <https://jmlr.org/papers/v13/bergstra12a.html>`__

    We use the QMC implementations in Scipy. For the details of the QMC algorithm,
    see the Scipy API references on `scipy.stats.qmc
    <https://scipy.github.io/devdocs/reference/stats.qmc.html>`__.

    .. note:
        If your search space contains categorical parameters, it samples the categorical
        parameters by its `independent_sampler` without using QMC algorithm.

    .. note::
        The search space of the sampler is determined by either previous trials in the study or
        the first trial that this sampler samples.

        If there are previous trials in the study, :class:`~optuna.samplers.QMCSampler` infers its
        search space using the trial which was created first in the study.

        Otherwise (if the study has no previous trials), :class:`~optuna.samplers.QMCSampler`
        samples the first trial using its `independent_sampler` and then infers the search space
        in the second trial.

        As mentioned above, the search space of the :class:`~optuna.samplers.QMCSampler` is
        determined by the first trial of the study. Once the search space is determined, it cannot
        be changed afterwards.

    Args:
        qmc_type:
            The type of QMC sequence to be sampled. This must be one of
            `"halton"` and `"sobol"`. Default is `"sobol"`.

            .. note::
                Sobol' sequence is designed to have low-discrepancy property when the number of
                samples is :math:`n=2^m` for each positive integer :math:`m`. When it is possible
                to pre-specify the number of trials suggested by `QMCSampler`, it is recommended
                that the number of trials should be set as power of two.

        scramble:
            If this option is :obj:`True`, scrambling (randomization) is applied to the QMC
            sequences.

        seed:
            A seed for ``QMCSampler``. This argument is used only when ``scramble`` is :obj:`True`.
            If this is :obj:`None`, the seed is initialized randomly. Default is :obj:`None`.

            .. note::
                When using multiple :class:`~optuna.samplers.QMCSampler`'s in parallel and/or
                distributed optimization, all the samplers must share the same seed when the
                `scrambling` is enabled. Otherwise, the low-discrepancy property of the samples
                will be degraded.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The first trial of the study and the parameters not contained in the
            relative search space are sampled by this sampler.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`~optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are sampled via an
            independent sampler in most cases, so no warning messages are emitted in such cases.

        warn_asynchronous_seeding:
            If this is :obj:`True`, a warning message is emitted when the scrambling
            (randomization) is applied to the QMC sequence and the random seed of the sampler is
            not set manually.

            .. note::
                When using parallel and/or distributed optimization without manually
                setting the seed, the seed is set randomly for each instances of
                :class:`~optuna.samplers.QMCSampler` for different workers, which ends up
                asynchronous seeding for multiple samplers used in the optimization.

            .. seealso::
                See parameter ``seed`` in :class:`~optuna.samplers.QMCSampler`.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.samplers.QMCSampler`.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x**2 + y


            sampler = optuna.samplers.QMCSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=8)

    """

    def __init__(
        self,
        *,
        qmc_type: str = "sobol",
        scramble: bool = False,  # default is False for simplicity in distributed environment.
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        warn_asynchronous_seeding: bool = True,
        warn_independent_sampling: bool = True,
    ) -> None:
        self._scramble = scramble
        self._seed = np.random.PCG64().random_raw() if seed is None else seed
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._initial_search_space: dict[str, BaseDistribution] | None = None
        self._warn_independent_sampling = warn_independent_sampling

        if qmc_type in ("halton", "sobol"):
            self._qmc_type = qmc_type
        else:
            message = (
                f'The `qmc_type`, "{qmc_type}", is not a valid. '
                'It must be one of "halton" and "sobol".'
            )
            raise ValueError(message)

        if seed is None and scramble and warn_asynchronous_seeding:
            # Sobol/Halton sequences without scrambling do not use seed.
            self._log_asynchronous_seeding()

    def reseed_rng(self) -> None:
        # We must not reseed the `self._seed` like below. Otherwise, workers will have different
        # seed under parallel execution because `self.reseed_rng()` is called when starting each
        # parallel executor.
        # >>> self._seed = np.random.MT19937().random_raw()

        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        if self._initial_search_space is not None:
            return self._initial_search_space

        past_trials = study._get_trials(deepcopy=False, states=_SUGGESTED_STATES, use_cache=True)
        # The initial trial is sampled by the independent sampler.
        if len(past_trials) == 0:
            return {}
        # If an initial trial was already made,
        # construct search_space of this sampler from the initial trial.
        first_trial = min(past_trials, key=lambda t: t.number)
        self._initial_search_space = self._infer_initial_search_space(first_trial)
        return self._initial_search_space

    def _infer_initial_search_space(self, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for param_name, distribution in trial.distributions.items():
            if isinstance(distribution, CategoricalDistribution):
                continue
            search_space[param_name] = distribution

        return search_space

    @staticmethod
    def _log_asynchronous_seeding() -> None:
        _logger.warning(
            "No seed is provided for `QMCSampler` and the seed is set randomly. "
            "If you are running multiple `QMCSampler`s in parallel and/or distributed "
            " environment, the same seed must be used in all samplers to ensure that resulting "
            "samples are taken from the same QMC sequence. "
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        _logger.warning(
            f"The parameter '{param_name}' in trial#{trial.number} is sampled independently "
            f"by using `{self._independent_sampler.__class__.__name__}` instead of `QMCSampler` "
            "(optimization performance may be degraded). "
            "`QMCSampler` does not support dynamic search space or `CategoricalDistribution`. "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `QMCSampler`, "
            "if this independent sampling is intended behavior."
        )

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if self._initial_search_space is not None:
            if self._warn_independent_sampling:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        sample = self._sample_qmc(study, search_space)
        trans = _SearchSpaceTransform(search_space)
        sample = trans.bounds[:, 0] + sample * (trans.bounds[:, 1] - trans.bounds[:, 0])
        return trans.untransform(sample[0, :])

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)

    def _sample_qmc(self, study: Study, search_space: dict[str, BaseDistribution]) -> np.ndarray:
        # Lazy import because the `scipy.stats.qmc` is slow to import.
        qmc_module = _LazyImport("scipy.stats.qmc")

        sample_id = self._find_sample_id(study)
        d = len(search_space)

        if self._qmc_type == "halton":
            qmc_engine = qmc_module.Halton(d, seed=self._seed, scramble=self._scramble)
        elif self._qmc_type == "sobol":
            # Sobol engine likely shares its internal state among threads.
            # Without threading.Lock, ValueError exceptions are raised in Sobol engine as discussed
            # in https://github.com/optuna/optunahub-registry/pull/168#pullrequestreview-2404054969
            with _threading_lock:
                qmc_engine = qmc_module.Sobol(d, seed=self._seed, scramble=self._scramble)
        else:
            raise ValueError("Invalid `qmc_type`")

        forward_size = sample_id  # `sample_id` starts from 0.
        # Skip fast_forward with forward_size==0 because Sobol doesn't support the case,
        # and fast_forward(0) doesn't affect sampling.
        if forward_size > 0:
            qmc_engine.fast_forward(forward_size)
        sample = qmc_engine.random(1)

        return sample

    def _find_sample_id(self, study: Study) -> int:
        qmc_id = ""
        qmc_id += self._qmc_type
        # Sobol/Halton sequences without scrambling do not use seed.
        if self._scramble:
            qmc_id += f" (scramble=True, seed={self._seed})"
        else:
            qmc_id += " (scramble=False)"
        key_qmc_id = qmc_id + "'s last sample id"

        # TODO(kstoneriv3): Here, we ideally assume that the following block is
        # an atomic transaction. Without such an assumption, the current implementation
        # only ensures that each `sample_id` is sampled at least once.
        system_attrs = study._storage.get_study_system_attrs(study._study_id)
        if key_qmc_id in system_attrs.keys():
            sample_id = system_attrs[key_qmc_id]
            sample_id += 1
        else:
            sample_id = 0
        study._storage.set_study_system_attr(study._study_id, key_qmc_id, sample_id)

        return sample_id
