from __future__ import annotations

import binascii
from collections.abc import Container
import math

import optuna
from optuna import logging
from optuna.pruners._base import BasePruner
from optuna.pruners._successive_halving import SuccessiveHalvingPruner
from optuna.trial._state import TrialState


_logger = logging.get_logger(__name__)


class HyperbandPruner(BasePruner):
    """Pruner using Hyperband.

    As SuccessiveHalving (SHA) requires the number of configurations
    :math:`n` as its hyperparameter.  For a given finite budget :math:`B`,
    all the configurations have the resources of :math:`B \\over n` on average.
    As you can see, there will be a trade-off of :math:`B` and :math:`B \\over n`.
    `Hyperband <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`__ attacks this trade-off
    by trying different :math:`n` values for a fixed budget.

    .. note::
        * In the Hyperband paper, the counterpart of :class:`~optuna.samplers.RandomSampler`
          is used.
        * Optuna uses :class:`~optuna.samplers.TPESampler` by default.
        * `The benchmark result
          <https://github.com/optuna/optuna/pull/828#issuecomment-575457360>`__
          shows that :class:`optuna.pruners.HyperbandPruner` supports both samplers.

    .. note::
        If you use ``HyperbandPruner`` with :class:`~optuna.samplers.TPESampler`,
        it's recommended to consider setting larger ``n_trials`` or ``timeout`` to make full use of
        the characteristics of :class:`~optuna.samplers.TPESampler`
        because :class:`~optuna.samplers.TPESampler` uses some (by default, :math:`10`)
        :class:`~optuna.trial.Trial`\\ s for its startup.

        As Hyperband runs multiple :class:`~optuna.pruners.SuccessiveHalvingPruner` and collects
        trials based on the current :class:`~optuna.trial.Trial`\\ 's bracket ID, each bracket
        needs to observe more than :math:`10` :class:`~optuna.trial.Trial`\\ s
        for :class:`~optuna.samplers.TPESampler` to adapt its search space.

        Thus, for example, if ``HyperbandPruner`` has :math:`4` pruners in it,
        at least :math:`4 \\times 10` trials are consumed for startup.

    .. note::
        Hyperband has several :class:`~optuna.pruners.SuccessiveHalvingPruner`\\ s. Each
        :class:`~optuna.pruners.SuccessiveHalvingPruner` is referred to as "bracket" in the
        original paper. The number of brackets is an important factor to control the early
        stopping behavior of Hyperband and is automatically determined by ``min_resource``,
        ``max_resource`` and ``reduction_factor`` as
        :math:`\\mathrm{The\\ number\\ of\\ brackets} =
        \\mathrm{floor}(\\log_{\\texttt{reduction}\\_\\texttt{factor}}
        (\\frac{\\texttt{max}\\_\\texttt{resource}}{\\texttt{min}\\_\\texttt{resource}})) + 1`.
        Please set ``reduction_factor`` so that the number of brackets is not too large (about 4 –
        6 in most use cases). Please see Section 3.6 of the `original paper
        <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`__ for the detail.

    Example:

        We minimize an objective function with Hyperband pruning algorithm.

        .. testcode::

            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            import optuna

            X, y = load_iris(return_X_y=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)
            classes = np.unique(y)
            n_train_iter = 100


            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_valid, y_valid)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return clf.score(X_valid, y_valid)


            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=1, max_resource=n_train_iter, reduction_factor=3
                ),
            )
            study.optimize(objective, n_trials=20)

    Args:
        min_resource:
            A parameter for specifying the minimum resource allocated to a trial noted as :math:`r`
            in the paper. A smaller :math:`r` will give a result faster, but a larger
            :math:`r` will give a better guarantee of successful judging between configurations.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        max_resource:
            A parameter for specifying the maximum resource allocated to a trial. :math:`R` in the
            paper corresponds to ``max_resource / min_resource``. This value represents and should
            match the maximum iteration steps (e.g., the number of epochs for neural networks).
            When this argument is "auto", the maximum resource is estimated according to the
            completed trials. The default value of this argument is "auto".

            .. note::
                With "auto", the maximum resource will be the largest step reported by
                :meth:`~optuna.trial.Trial.report` in the first, or one of the first if trained in
                parallel, completed trial. No trials will be pruned until the maximum resource is
                determined.

            .. note::
                If the step of the last intermediate value may change with each trial, please
                manually specify the maximum possible step to ``max_resource``.
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials noted as
            :math:`\\eta` in the paper.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        bootstrap_count:
            Parameter specifying the number of trials required in a rung before any trial can be
            promoted. Incompatible with ``max_resource`` is ``"auto"``.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
    """

    def __init__(
        self,
        min_resource: int = 1,
        max_resource: str | int = "auto",
        reduction_factor: int = 3,
        bootstrap_count: int = 0,
    ) -> None:
        self._min_resource = min_resource
        self._max_resource = max_resource
        self._reduction_factor = reduction_factor
        self._pruners: list[SuccessiveHalvingPruner] = []
        self._bootstrap_count = bootstrap_count
        self._total_trial_allocation_budget = 0
        self._trial_allocation_budgets: list[int] = []
        self._n_brackets: int | None = None

        if not isinstance(self._max_resource, int) and self._max_resource != "auto":
            raise ValueError(
                "The 'max_resource' should be integer or 'auto'. "
                "But max_resource = {}".format(self._max_resource)
            )

        if self._bootstrap_count > 0 and self._max_resource == "auto":
            raise ValueError(
                "bootstrap_count > 0 and max_resource == 'auto' "
                "are mutually incompatible, bootstrap_count is {}".format(self._bootstrap_count)
            )

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        if len(self._pruners) == 0:
            self._try_initialization(study)
            if len(self._pruners) == 0:
                return False

        bracket_id = self._get_bracket_id(study, trial)
        _logger.debug("{}th bracket is selected".format(bracket_id))
        bracket_study = self._create_bracket_study(study, bracket_id)
        return self._pruners[bracket_id].prune(bracket_study, trial)

    def _try_initialization(self, study: "optuna.study.Study") -> None:
        if self._max_resource == "auto":
            trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            n_steps = [t.last_step for t in trials if t.last_step is not None]

            if not n_steps:
                return

            self._max_resource = max(n_steps) + 1

        assert isinstance(self._max_resource, int)

        if self._n_brackets is None:
            # In the original paper http://www.jmlr.org/papers/volume18/16-558/16-558.pdf, the
            # inputs of Hyperband are `R`: max resource and `\eta`: reduction factor. The
            # number of brackets (this is referred as `s_{max} + 1` in the paper) is calculated
            # by s_{max} + 1 = \floor{\log_{\eta} (R)} + 1 in Algorithm 1 of the original paper.
            # In this implementation, we combine this formula and that of ASHA paper
            # https://arxiv.org/abs/1502.07943 as
            # `n_brackets = floor(log_{reduction_factor}(max_resource / min_resource)) + 1`
            self._n_brackets = (
                math.floor(
                    math.log(self._max_resource / self._min_resource, self._reduction_factor)
                )
                + 1
            )

        _logger.debug("Hyperband has {} brackets".format(self._n_brackets))

        for bracket_id in range(self._n_brackets):
            trial_allocation_budget = self._calculate_trial_allocation_budget(bracket_id)
            self._total_trial_allocation_budget += trial_allocation_budget
            self._trial_allocation_budgets.append(trial_allocation_budget)

            pruner = SuccessiveHalvingPruner(
                min_resource=self._min_resource,
                reduction_factor=self._reduction_factor,
                min_early_stopping_rate=bracket_id,
                bootstrap_count=self._bootstrap_count,
            )
            self._pruners.append(pruner)

    def _calculate_trial_allocation_budget(self, bracket_id: int) -> int:
        """Compute the trial allocated budget for a bracket of ``bracket_id``.

        In the `original paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`, the
        number of trials per one bracket is referred as ``n`` in Algorithm 1. Since we do not know
        the total number of trials in the leaning scheme of Optuna, we calculate the ratio of the
        number of trials here instead.
        """

        assert self._n_brackets is not None
        s = self._n_brackets - 1 - bracket_id
        return math.ceil(self._n_brackets * (self._reduction_factor**s) / (s + 1))

    def _get_bracket_id(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> int:
        """Compute the index of bracket for a trial of ``trial_number``.

        The index of a bracket is noted as :math:`s` in
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`__.
        """

        if len(self._pruners) == 0:
            return 0

        assert self._n_brackets is not None
        n = (
            binascii.crc32("{}_{}".format(study.study_name, trial.number).encode())
            % self._total_trial_allocation_budget
        )
        for bracket_id in range(self._n_brackets):
            n -= self._trial_allocation_budgets[bracket_id]
            if n < 0:
                return bracket_id

        assert False, "This line should be unreachable."

    def _create_bracket_study(
        self, study: "optuna.study.Study", bracket_id: int
    ) -> "optuna.study.Study":
        # This class is assumed to be passed to
        # `SuccessiveHalvingPruner.prune` in which `get_trials`,
        # `direction`, and `storage` are used.
        # But for safety, prohibit the other attributes explicitly.
        class _BracketStudy(optuna.study.Study):
            _VALID_ATTRS = (
                "get_trials",
                "_get_trials",
                "directions",
                "direction",
                "_directions",
                "_storage",
                "_study_id",
                "pruner",
                "study_name",
                "_bracket_id",
                "sampler",
                "trials",
                "_is_multi_objective",
                "stop",
                "_study",
                "_thread_local",
            )

            def __init__(
                self, study: "optuna.study.Study", pruner: HyperbandPruner, bracket_id: int
            ) -> None:
                super().__init__(
                    study_name=study.study_name,
                    storage=study._storage,
                    sampler=study.sampler,
                    pruner=pruner,
                )
                self._study = study
                self._bracket_id = bracket_id

            def get_trials(
                self,
                deepcopy: bool = True,
                states: Container[TrialState] | None = None,
            ) -> list["optuna.trial.FrozenTrial"]:
                trials = super()._get_trials(deepcopy=deepcopy, states=states)
                pruner = self.pruner
                assert isinstance(pruner, HyperbandPruner)
                return [t for t in trials if pruner._get_bracket_id(self, t) == self._bracket_id]

            def stop(self) -> None:
                # `stop` should stop the original study's optimization loop instead of
                # `_BracketStudy`.
                self._study.stop()

            def __getattribute__(self, attr_name):  # type: ignore
                if attr_name not in _BracketStudy._VALID_ATTRS:
                    raise AttributeError(
                        "_BracketStudy does not have attribute of '{}'".format(attr_name)
                    )
                else:
                    return object.__getattribute__(self, attr_name)

        return _BracketStudy(study, self, bracket_id)
