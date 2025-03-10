from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import copy
import math
import pickle
from typing import Any
from typing import cast
from typing import NamedTuple
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

import optuna
from optuna import logging
from optuna._experimental import warn_experimental_argument
from optuna._imports import _LazyImport
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import cmaes

    CmaClass = Union[cmaes.CMA, cmaes.SepCMA, cmaes.CMAwM]
else:
    cmaes = _LazyImport("cmaes")

_logger = logging.get_logger(__name__)

_EPS = 1e-10
# The value of system_attrs must be less than 2046 characters on RDBStorage.
_SYSTEM_ATTR_MAX_LENGTH = 2045


class _CmaEsAttrKeys(NamedTuple):
    optimizer: Callable[[int], str]
    generation: Callable[[int], str]
    popsize: Callable[[], str]
    n_restarts: Callable[[], str]
    n_restarts_with_large: str
    poptype: str
    small_n_eval: str
    large_n_eval: str


class CmaEsSampler(BaseSampler):
    """A sampler using `cmaes <https://github.com/CyberAgentAILab/cmaes>`__ as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.samplers.CmaEsSampler`.

        .. code-block:: console

           $ pip install cmaes

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x**2 + y


            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=20)

    Please note that this sampler does not support CategoricalDistribution.
    However, :class:`~optuna.distributions.FloatDistribution` with ``step``,
    (:func:`~optuna.trial.Trial.suggest_float`) and
    :class:`~optuna.distributions.IntDistribution` (:func:`~optuna.trial.Trial.suggest_int`)
    are supported.

    If your search space contains categorical parameters, I recommend you
    to use :class:`~optuna.samplers.TPESampler` instead.
    Furthermore, there is room for performance improvements in parallel
    optimization settings. This sampler cannot use some trials for updating
    the parameters of multivariate normal distribution.

    For further information about CMA-ES algorithm, please refer to the following papers:

    - `N. Hansen, The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.
      <https://arxiv.org/abs/1604.00772>`__
    - `A. Auger and N. Hansen. A restart CMA evolution strategy with increasing population
      size. In Proceedings of the IEEE Congress on Evolutionary Computation (CEC 2005),
      pages 1769–1776. IEEE Press, 2005. <https://doi.org/10.1109/CEC.2005.1554902>`__
    - `N. Hansen. Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed.
      GECCO Workshop, 2009. <https://doi.org/10.1145/1570256.1570333>`__
    - `Raymond Ros, Nikolaus Hansen. A Simple Modification in CMA-ES Achieving Linear Time and
      Space Complexity. 10th International Conference on Parallel Problem Solving From Nature,
      Sep 2008, Dortmund, Germany. inria-00287367. <https://doi.org/10.1007/978-3-540-87700-4_30>`__
    - `Masahiro Nomura, Shuhei Watanabe, Youhei Akimoto, Yoshihiko Ozaki, Masaki Onishi.
      Warm Starting CMA-ES for Hyperparameter Optimization, AAAI. 2021.
      <https://doi.org/10.1609/aaai.v35i10.17109>`__
    - `R. Hamano, S. Saito, M. Nomura, S. Shirakawa. CMA-ES with Margin: Lower-Bounding Marginal
      Probability for Mixed-Integer Black-Box Optimization, GECCO. 2022.
      <https://doi.org/10.1145/3512290.3528827>`__
    - `M. Nomura, Y. Akimoto, I. Ono. CMA-ES with Learning Rate Adaptation: Can CMA-ES with
      Default Population Size Solve Multimodal and Noisy Problems?, GECCO. 2023.
      <https://doi.org/10.1145/3583131.3590358>`__

    .. seealso::
        You can also use `optuna_integration.PyCmaSampler <https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.PyCmaSampler.html#optuna_integration.PyCmaSampler>`__ which is a sampler using cma
        library as the backend.

    Args:

        x0:
            A dictionary of an initial parameter values for CMA-ES. By default, the mean of ``low``
            and ``high`` for each distribution is used. Note that ``x0`` is sampled uniformly
            within the search space domain for each restart if you specify ``restart_strategy``
            argument.

        sigma0:
            Initial standard deviation of CMA-ES. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space.

        seed:
            A random seed for CMA-ES.

        n_startup_trials:
            The independent sampling is used instead of the CMA-ES algorithm until the given number
            of trials finish in the same study.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.CmaEsSampler` is determined by
            :func:`~optuna.search_space.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.

        restart_strategy:
            Strategy for restarting CMA-ES optimization when converges to a local minimum.
            If :obj:`None` is given, CMA-ES will not restart (default).
            If 'ipop' is given, CMA-ES will restart with increasing population size.
            if 'bipop' is given, CMA-ES will restart with the population size
            increased or decreased.
            Please see also ``inc_popsize`` parameter.

            .. note::
                Added in v2.1.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.1.0.

        popsize:
            A population size of CMA-ES. When ``restart_strategy = 'ipop'``
            or ``restart_strategy = 'bipop'`` is specified,
            this is used as the initial population size.

        inc_popsize:
            Multiplier for increasing population size before each restart.
            This argument will be used when ``restart_strategy = 'ipop'``
            or ``restart_strategy = 'bipop'`` is specified.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.

            .. note::
                Added in v2.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.0.0.

            .. note::
                It is suggested to set this flag :obj:`False` when the
                :class:`~optuna.pruners.MedianPruner` is used. On the other hand, it is suggested
                to set this flag :obj:`True` when the :class:`~optuna.pruners.HyperbandPruner` is
                used. Please see `the benchmark result
                <https://github.com/optuna/optuna/pull/1229>`__ for the details.

        use_separable_cma:
            If this is :obj:`True`, the covariance matrix is constrained to be diagonal.
            Due to reduce the model complexity, the learning rate for the covariance matrix
            is increased. Consequently, this algorithm outperforms CMA-ES on separable functions.

            .. note::
                Added in v2.6.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.6.0.

        with_margin:
            If this is :obj:`True`, CMA-ES with margin is used. This algorithm prevents samples in
            each discrete distribution (:class:`~optuna.distributions.FloatDistribution` with
            ``step`` and :class:`~optuna.distributions.IntDistribution`) from being fixed to a single
            point.
            Currently, this option cannot be used with ``use_separable_cma=True``.

            .. note::
                Added in v3.1.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v3.1.0.

        lr_adapt:
            If this is :obj:`True`, CMA-ES with learning rate adaptation is used.
            This algorithm focuses on working well on multimodal and/or noisy problems
            with default settings.
            Currently, this option cannot be used with ``use_separable_cma=True`` or
            ``with_margin=True``.

            .. note::
                Added in v3.3.0 or later, as an experimental feature.
                The interface may change in newer versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v3.3.0.

        source_trials:
            This option is for Warm Starting CMA-ES, a method to transfer prior knowledge on
            similar HPO tasks through the initialization of CMA-ES. This method estimates a
            promising distribution from ``source_trials`` and generates the parameter of
            multivariate gaussian distribution. Please note that it is prohibited to use
            ``x0``, ``sigma0``, or ``use_separable_cma`` argument together.

            .. note::
                Added in v2.6.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.6.0.

    """  # NOQA: E501

    def __init__(
        self,
        x0: dict[str, Any] | None = None,
        sigma0: float | None = None,
        n_startup_trials: int = 1,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        seed: int | None = None,
        *,
        consider_pruned_trials: bool = False,
        restart_strategy: str | None = None,
        popsize: int | None = None,
        inc_popsize: int = 2,
        use_separable_cma: bool = False,
        with_margin: bool = False,
        lr_adapt: bool = False,
        source_trials: list[FrozenTrial] | None = None,
    ) -> None:
        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._cma_rng = LazyRandomState(seed)
        self._search_space = IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials
        self._restart_strategy = restart_strategy
        self._initial_popsize = popsize
        self._inc_popsize = inc_popsize
        self._use_separable_cma = use_separable_cma
        self._with_margin = with_margin
        self._lr_adapt = lr_adapt
        self._source_trials = source_trials

        if self._restart_strategy:
            warn_experimental_argument("restart_strategy")

        if self._consider_pruned_trials:
            warn_experimental_argument("consider_pruned_trials")

        if self._use_separable_cma:
            warn_experimental_argument("use_separable_cma")

        if self._source_trials is not None:
            warn_experimental_argument("source_trials")

        if self._with_margin:
            warn_experimental_argument("with_margin")

        if self._lr_adapt:
            warn_experimental_argument("lr_adapt")

        if source_trials is not None and (x0 is not None or sigma0 is not None):
            raise ValueError(
                "It is prohibited to pass `source_trials` argument when "
                "x0 or sigma0 is specified."
            )

        # TODO(c-bata): Support WS-sep-CMA-ES.
        if source_trials is not None and use_separable_cma:
            raise ValueError(
                "It is prohibited to pass `source_trials` argument when using separable CMA-ES."
            )

        if lr_adapt and (use_separable_cma or with_margin):
            raise ValueError(
                "It is prohibited to pass `use_separable_cma` or `with_margin` argument when "
                "using `lr_adapt`."
            )

        if restart_strategy not in (
            "ipop",
            "bipop",
            None,
        ):
            raise ValueError(
                "restart_strategy={} is unsupported. "
                "Please specify: 'ipop', 'bipop', or None.".format(restart_strategy)
            )

        # TODO(knshnb): Support sep-CMA-ES with margin.
        if self._use_separable_cma and self._with_margin:
            raise ValueError(
                "Currently, we do not support `use_separable_cma=True` and `with_margin=True`."
            )

    def reseed_rng(self) -> None:
        # _cma_rng doesn't require reseeding because the relative sampling reseeds in each trial.
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            if not isinstance(distribution, (FloatDistribution, IntDistribution)):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = self._get_trials(study)
        if len(completed_trials) < self._n_startup_trials:
            return {}

        if len(search_space) == 1:
            if self._warn_independent_sampling:
                _logger.warning(
                    "`CmaEsSampler` only supports two or more dimensional continuous "
                    "search space. `{}` is used instead of `CmaEsSampler`.".format(
                        self._independent_sampler.__class__.__name__
                    )
                )
                self._warn_independent_sampling = False
            return {}

        # When `with_margin=True`, bounds in discrete dimensions are handled inside `CMAwM`.
        trans = _SearchSpaceTransform(
            search_space, transform_step=not self._with_margin, transform_0_1=True
        )

        if self._initial_popsize is None:
            self._initial_popsize = 4 + math.floor(3 * math.log(len(trans.bounds)))

        popsize: int = self._initial_popsize
        n_restarts: int = 0
        n_restarts_with_large: int = 0
        poptype: str = "small"
        small_n_eval: int = 0
        large_n_eval: int = 0
        if len(completed_trials) != 0:
            latest_trial = completed_trials[-1]

            popsize_attr_key = self._attr_keys.popsize()
            if popsize_attr_key in latest_trial.system_attrs:
                popsize = latest_trial.system_attrs[popsize_attr_key]
            else:
                popsize = self._initial_popsize

            n_restarts_attr_key = self._attr_keys.n_restarts()
            n_restarts = latest_trial.system_attrs.get(n_restarts_attr_key, 0)
            n_restarts_with_large = latest_trial.system_attrs.get(
                self._attr_keys.n_restarts_with_large, 0
            )
            poptype = latest_trial.system_attrs.get(self._attr_keys.poptype, "small")
            small_n_eval = latest_trial.system_attrs.get(self._attr_keys.small_n_eval, 0)
            large_n_eval = latest_trial.system_attrs.get(self._attr_keys.large_n_eval, 0)

        optimizer = self._restore_optimizer(completed_trials, n_restarts)
        if optimizer is None:
            optimizer = self._init_optimizer(
                trans, study.direction, population_size=self._initial_popsize
            )

        if optimizer.dim != len(trans.bounds):
            if self._warn_independent_sampling:
                _logger.warning(
                    "`CmaEsSampler` does not support dynamic search space. "
                    "`{}` is used instead of `CmaEsSampler`.".format(
                        self._independent_sampler.__class__.__name__
                    )
                )
                self._warn_independent_sampling = False
            return {}

        # TODO(c-bata): Reduce the number of wasted trials during parallel optimization.
        # See https://github.com/optuna/optuna/pull/920#discussion_r385114002 for details.
        solution_trials = self._get_solution_trials(
            completed_trials, optimizer.generation, n_restarts
        )

        if len(solution_trials) >= popsize:
            solutions: list[tuple[np.ndarray, float]] = []
            for t in solution_trials[:popsize]:
                assert t.value is not None, "completed trials must have a value"
                if isinstance(optimizer, cmaes.CMAwM):
                    x = np.array(t.system_attrs["x_for_tell"])
                else:
                    x = trans.transform(t.params)
                y = t.value if study.direction == StudyDirection.MINIMIZE else -t.value
                solutions.append((x, y))

            optimizer.tell(solutions)

            if self._restart_strategy == "ipop" and optimizer.should_stop():
                n_restarts += 1
                popsize = popsize * self._inc_popsize
                optimizer = self._init_optimizer(
                    trans, study.direction, population_size=popsize, randomize_start_point=True
                )

            if self._restart_strategy == "bipop" and optimizer.should_stop():
                n_restarts += 1

                n_eval = popsize * optimizer.generation
                if poptype == "small":
                    small_n_eval += n_eval
                else:  # poptype == "large"
                    large_n_eval += n_eval

                if small_n_eval < large_n_eval:
                    poptype = "small"
                    popsize_multiplier = self._inc_popsize**n_restarts_with_large
                    popsize = math.floor(
                        self._initial_popsize
                        * popsize_multiplier ** (self._cma_rng.rng.uniform() ** 2)
                    )
                else:
                    poptype = "large"
                    n_restarts_with_large += 1
                    popsize = self._initial_popsize * (self._inc_popsize**n_restarts_with_large)

                optimizer = self._init_optimizer(
                    trans, study.direction, population_size=popsize, randomize_start_point=True
                )

            # Store optimizer.
            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = self._split_optimizer_str(optimizer_str, n_restarts)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

        # Caution: optimizer should update its seed value.
        seed = self._cma_rng.rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)
        if isinstance(optimizer, cmaes.CMAwM):
            params, x_for_tell = optimizer.ask()
            study._storage.set_trial_system_attr(
                trial._trial_id, "x_for_tell", x_for_tell.tolist()
            )
        else:
            params = optimizer.ask()

        generation_attr_key = self._attr_keys.generation(n_restarts)
        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )
        popsize_attr_key = self._attr_keys.popsize()
        study._storage.set_trial_system_attr(trial._trial_id, popsize_attr_key, popsize)
        n_restarts_attr_key = self._attr_keys.n_restarts()
        study._storage.set_trial_system_attr(trial._trial_id, n_restarts_attr_key, n_restarts)
        study._storage.set_trial_system_attr(
            trial._trial_id, self._attr_keys.n_restarts_with_large, n_restarts_with_large
        )
        study._storage.set_trial_system_attr(trial._trial_id, self._attr_keys.poptype, poptype)
        study._storage.set_trial_system_attr(
            trial._trial_id, self._attr_keys.small_n_eval, small_n_eval
        )
        study._storage.set_trial_system_attr(
            trial._trial_id, self._attr_keys.large_n_eval, large_n_eval
        )

        external_values = trans.untransform(params)

        return external_values

    @property
    def _attr_keys(self) -> _CmaEsAttrKeys:
        if self._use_separable_cma:
            attr_prefix = "sepcma:"
        elif self._with_margin:
            attr_prefix = "cmawm:"
        else:
            attr_prefix = "cma:"

        def optimizer_key_template(restart: int) -> str:
            if self._restart_strategy is None:
                return attr_prefix + "optimizer"
            else:
                return attr_prefix + "{}:restart_{}:optimizer".format(
                    self._restart_strategy, restart
                )

        def generation_attr_key_template(restart: int) -> str:
            if self._restart_strategy is None:
                return attr_prefix + "generation"
            else:
                return attr_prefix + "{}:restart_{}:generation".format(
                    self._restart_strategy, restart
                )

        def popsize_attr_key_template() -> str:
            if self._restart_strategy is None:
                return attr_prefix + "popsize"
            else:
                return attr_prefix + "{}:popsize".format(self._restart_strategy)

        def n_restarts_attr_key_template() -> str:
            if self._restart_strategy is None:
                return attr_prefix + "n_restarts"
            else:
                return attr_prefix + "{}:n_restarts".format(self._restart_strategy)

        return _CmaEsAttrKeys(
            optimizer_key_template,
            generation_attr_key_template,
            popsize_attr_key_template,
            n_restarts_attr_key_template,
            attr_prefix + "n_restarts_with_large",
            attr_prefix + "poptype",
            attr_prefix + "small_n_eval",
            attr_prefix + "large_n_eval",
        )

    def _concat_optimizer_attrs(self, optimizer_attrs: dict[str, str], n_restarts: int = 0) -> str:
        return "".join(
            optimizer_attrs["{}:{}".format(self._attr_keys.optimizer(n_restarts), i)]
            for i in range(len(optimizer_attrs))
        )

    def _split_optimizer_str(self, optimizer_str: str, n_restarts: int = 0) -> dict[str, str]:
        optimizer_len = len(optimizer_str)
        attrs = {}
        for i in range(math.ceil(optimizer_len / _SYSTEM_ATTR_MAX_LENGTH)):
            start = i * _SYSTEM_ATTR_MAX_LENGTH
            end = min((i + 1) * _SYSTEM_ATTR_MAX_LENGTH, optimizer_len)
            attrs["{}:{}".format(self._attr_keys.optimizer(n_restarts), i)] = optimizer_str[
                start:end
            ]
        return attrs

    def _restore_optimizer(
        self,
        completed_trials: "list[optuna.trial.FrozenTrial]",
        n_restarts: int = 0,
    ) -> "CmaClass" | None:
        # Restore a previous CMA object.
        for trial in reversed(completed_trials):
            optimizer_attrs = {
                key: value
                for key, value in trial.system_attrs.items()
                if key.startswith(self._attr_keys.optimizer(n_restarts))
            }
            if len(optimizer_attrs) == 0:
                continue

            optimizer_str = self._concat_optimizer_attrs(optimizer_attrs, n_restarts)
            return pickle.loads(bytes.fromhex(optimizer_str))
        return None

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        direction: StudyDirection,
        population_size: int | None = None,
        randomize_start_point: bool = False,
    ) -> "CmaClass":
        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if self._source_trials is None:
            if randomize_start_point:
                mean = lower_bounds + (upper_bounds - lower_bounds) * self._cma_rng.rng.rand(
                    n_dimension
                )
            elif self._x0 is None:
                mean = lower_bounds + (upper_bounds - lower_bounds) / 2
            else:
                # `self._x0` is external representations.
                mean = trans.transform(self._x0)

            if self._sigma0 is None:
                sigma0 = np.min((upper_bounds - lower_bounds) / 6)
            else:
                sigma0 = self._sigma0

            cov = None
        else:
            expected_states = [TrialState.COMPLETE]
            if self._consider_pruned_trials:
                expected_states.append(TrialState.PRUNED)

            # TODO(c-bata): Filter parameters by their values instead of checking search space.
            sign = 1 if direction == StudyDirection.MINIMIZE else -1
            source_solutions = [
                (trans.transform(t.params), sign * cast(float, t.value))
                for t in self._source_trials
                if t.state in expected_states
                and _is_compatible_search_space(trans, t.distributions)
            ]
            if len(source_solutions) == 0:
                raise ValueError("No compatible source_trials")

            # TODO(c-bata): Add options to change prior parameters (alpha and gamma).
            mean, sigma0, cov = cmaes.get_warm_start_mgd(source_solutions)

        # Avoid ZeroDivisionError in cmaes.
        sigma0 = max(sigma0, _EPS)

        if self._use_separable_cma:
            return cmaes.SepCMA(
                mean=mean,
                sigma=sigma0,
                bounds=trans.bounds,
                seed=self._cma_rng.rng.randint(1, 2**31 - 2),
                n_max_resampling=10 * n_dimension,
                population_size=population_size,
            )

        if self._with_margin:
            steps = np.empty(len(trans._search_space), dtype=float)
            for i, dist in enumerate(trans._search_space.values()):
                assert isinstance(dist, (IntDistribution, FloatDistribution))
                # Set step 0.0 for continuous search space.
                if dist.step is None or dist.log:
                    steps[i] = 0.0
                elif dist.low == dist.high:
                    steps[i] = 1.0
                else:
                    steps[i] = dist.step / (dist.high - dist.low)

            return cmaes.CMAwM(
                mean=mean,
                sigma=sigma0,
                bounds=trans.bounds,
                steps=steps,
                cov=cov,
                seed=self._cma_rng.rng.randint(1, 2**31 - 2),
                n_max_resampling=10 * n_dimension,
                population_size=population_size,
            )

        return cmaes.CMA(
            mean=mean,
            sigma=sigma0,
            cov=cov,
            bounds=trans.bounds,
            seed=self._cma_rng.rng.randint(1, 2**31 - 2),
            n_max_resampling=10 * n_dimension,
            population_size=population_size,
            lr_adapt=self._lr_adapt,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)

        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        _logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `CmaEsSampler` "
            "(optimization performance may be degraded). "
            "`CmaEsSampler` does not support dynamic search space or `CategoricalDistribution`. "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `CmaEsSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )

    def _get_trials(self, study: "optuna.Study") -> list[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                # We rewrite the value of the trial `t` for sampling, so we need a deepcopy.
                copied_t = copy.deepcopy(t)
                copied_t.value = value
                complete_trials.append(copied_t)
        return complete_trials

    def _get_solution_trials(
        self, trials: list[FrozenTrial], generation: int, n_restarts: int
    ) -> list[FrozenTrial]:
        generation_attr_key = self._attr_keys.generation(n_restarts)
        return [t for t in trials if generation == t.system_attrs.get(generation_attr_key, -1)]

    def before_trial(self, study: optuna.Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


def _is_compatible_search_space(
    trans: _SearchSpaceTransform, search_space: dict[str, BaseDistribution]
) -> bool:
    intersection_size = len(set(trans._search_space.keys()).intersection(search_space.keys()))
    return intersection_size == len(trans._search_space) == len(search_space)
