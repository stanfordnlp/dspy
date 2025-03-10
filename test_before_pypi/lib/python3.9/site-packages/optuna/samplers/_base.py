from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


class BaseSampler(abc.ABC):
    """Base class for samplers.

    Optuna combines two types of sampling strategies, which are called *relative sampling* and
    *independent sampling*.

    *The relative sampling* determines values of multiple parameters simultaneously so that
    sampling algorithms can use relationship between parameters (e.g., correlation).
    Target parameters of the relative sampling are described in a relative search space, which
    is determined by :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.

    *The independent sampling* determines a value of a single parameter without considering any
    relationship between parameters. Target parameters of the independent sampling are the
    parameters not described in the relative search space.

    More specifically, parameters are sampled by the following procedure.
    At the beginning of a trial, :meth:`~optuna.samplers.BaseSampler.infer_relative_search_space`
    is called to determine the relative search space for the trial.
    During the execution of the objective function,
    :meth:`~optuna.samplers.BaseSampler.sample_relative` is called only once
    when sampling the parameters belonging to the relative search space for the first time.
    :meth:`~optuna.samplers.BaseSampler.sample_independent` is used to sample
    parameters that don't belong to the relative search space.

    The following figure depicts the lifetime of a trial and how the above three methods are
    called in the trial.

    .. image:: ../../../../image/sampling-sequence.png

    |

    """

    def __str__(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        """Infer the search space that will be used by relative sampling in the target trial.

        This method is called right before :func:`~optuna.samplers.BaseSampler.sample_relative`
        method, and the search space returned by this method is passed to it. The parameters not
        contained in the search space will be sampled by using
        :func:`~optuna.samplers.BaseSampler.sample_independent` method.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
                Take a copy before modifying this object.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        .. seealso::
            Please refer to :func:`~optuna.search_space.intersection_search_space` as an
            implementation of :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        """Sample parameters in a given search space.

        This method is called once at the beginning of each trial, i.e., right before the
        evaluation of the objective function. This method is suitable for sampling algorithms
        that use relationship between parameters such as Gaussian Process and CMA-ES.

        .. note::
                The failed trials are ignored by any build-in samplers when they sample new
                parameters. Thus, failed trials are regarded as deleted in the samplers'
                perspective.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
                Take a copy before modifying this object.
            search_space:
                The search space returned by
                :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.

        Returns:
            A dictionary containing the parameter names and the values.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a parameter for a given distribution.

        This method is called only for the parameters not contained in the search space returned
        by :func:`~optuna.samplers.BaseSampler.sample_relative` method. This method is suitable
        for sampling algorithms that do not use relationship between parameters such as random
        sampling and TPE.

        .. note::
                The failed trials are ignored by any build-in samplers when they sample new
                parameters. Thus, failed trials are regarded as deleted in the samplers'
                perspective.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
                Take a copy before modifying this object.
            param_name:
                Name of the sampled parameter.
            param_distribution:
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            A parameter value.

        """

        raise NotImplementedError

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        """Trial pre-processing.

        This method is called before the objective function is called and right after the trial is
        instantiated. More precisely, this method is called during trial initialization, just
        before the :func:`~optuna.samplers.BaseSampler.infer_relative_search_space` call. In other
        words, it is responsible for pre-processing that should be done before inferring the search
        space.

        .. note::
            Added in v3.3.0 as an experimental feature. The interface may change in newer versions
            without prior notice. See https://github.com/optuna/optuna/releases/tag/v3.3.0.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
        """

        pass

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """Trial post-processing.

        This method is called after the objective function returns and right before the trial is
        finished and its state is stored.

        .. note::
            Added in v2.4.0 as an experimental feature. The interface may change in newer versions
            without prior notice. See https://github.com/optuna/optuna/releases/tag/v2.4.0.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
                Take a copy before modifying this object.
            state:
                Resulting trial state.
            values:
                Resulting trial values. Guaranteed to not be :obj:`None` if trial succeeded.

        """

        pass

    def reseed_rng(self) -> None:
        """Reseed sampler's random number generator.

        This method is called by the :class:`~optuna.study.Study` instance if trials are executed
        in parallel with the option ``n_jobs>1``. In that case, the sampler instance will be
        replicated including the state of the random number generator, and they may suggest the
        same values. To prevent this issue, this method assigns a different seed to each random
        number generator.
        """

        pass

    def _raise_error_if_multi_objective(self, study: Study) -> None:
        if study._is_multi_objective():
            raise ValueError(
                "If the study is being used for multi-objective optimization, "
                f"{self.__class__.__name__} cannot be used."
            )


_CONSTRAINTS_KEY = "constraints"


def _process_constraints_after_trial(
    constraints_func: Callable[[FrozenTrial], Sequence[float]],
    study: Study,
    trial: FrozenTrial,
    state: TrialState,
) -> None:
    if state not in [TrialState.COMPLETE, TrialState.PRUNED]:
        return

    constraints = None
    try:
        con = constraints_func(trial)
        if np.any(np.isnan(con)):
            raise ValueError("Constraint values cannot be NaN.")
        if not isinstance(con, (tuple, list)):
            warnings.warn(
                f"Constraints should be a sequence of floats but got {type(con).__name__}."
            )
        constraints = tuple(con)
    finally:
        assert constraints is None or isinstance(constraints, tuple)

        study._storage.set_trial_system_attr(
            trial._trial_id,
            _CONSTRAINTS_KEY,
            constraints,
        )
