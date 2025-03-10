from __future__ import annotations

from collections import UserDict
from collections.abc import Sequence
import copy
import datetime
from typing import Any
from typing import overload
import warnings

import optuna
from optuna import distributions
from optuna import logging
from optuna import pruners
from optuna._convert_positional_args import convert_positional_args
from optuna._deprecated import deprecated_func
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import FrozenTrial
from optuna.trial._base import _SUGGEST_INT_POSITIONAL_ARGS
from optuna.trial._base import BaseTrial


_logger = logging.get_logger(__name__)
_suggest_deprecated_msg = "Use suggest_float{args} instead."


class Trial(BaseTrial):
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that the direct use of this constructor is not recommended.
    This object is seamlessly instantiated and passed to the objective function behind
    the :func:`optuna.study.Study.optimize()` method; hence library users do not care about
    instantiation of this object.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        trial_id:
            A trial ID that is automatically generated.

    """

    def __init__(self, study: "optuna.study.Study", trial_id: int) -> None:
        self.study = study
        self._trial_id = trial_id

        self.storage = self.study._storage

        self._cached_frozen_trial = self.storage.get_trial(self._trial_id)
        study = pruners._filter_study(self.study, self._cached_frozen_trial)

        self.study.sampler.before_trial(study, self._cached_frozen_trial)

        self.relative_search_space = self.study.sampler.infer_relative_search_space(
            study, self._cached_frozen_trial
        )
        self._relative_params: dict[str, Any] | None = None
        self._fixed_params = self._cached_frozen_trial.system_attrs.get("fixed_params", {})

    @property
    def relative_params(self) -> dict[str, Any]:
        if self._relative_params is None:
            study = pruners._filter_study(self.study, self._cached_frozen_trial)
            self._relative_params = self.study.sampler.sample_relative(
                study, self._cached_frozen_trial, self.relative_search_space
            )
        return self._relative_params

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Suggest a value for the floating point parameter.

        Example:

            Suggest a momentum, learning rate and scaling factor of learning rate
            for neural network training.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.neural_network import MLPClassifier

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)


                def objective(trial):
                    momentum = trial.suggest_float("momentum", 0.0, 1.0)
                    learning_rate_init = trial.suggest_float(
                        "learning_rate_init", 1e-5, 1e-3, log=True
                    )
                    power_t = trial.suggest_float("power_t", 0.2, 0.8, step=0.1)
                    clf = MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        momentum=momentum,
                        learning_rate_init=learning_rate_init,
                        solver="sgd",
                        random_state=0,
                        power_t=power_t,
                    )
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)


                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=3)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
                ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,
                ``low`` must be larger than 0.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
                ``high`` must be greater than or equal to ``low``.
            step:
                A step of discretization.

                .. note::
                    The ``step`` and ``log`` arguments cannot be used at the same time. To set
                    the ``step`` argument to a float number, set the ``log`` argument to
                    :obj:`False`.
            log:
                A flag to sample the value from the log domain or not.
                If ``log`` is true, the value is sampled from the range in the log domain.
                Otherwise, the value is sampled from the range in the linear domain.

                .. note::
                    The ``step`` and ``log`` arguments cannot be used at the same time. To set
                    the ``log`` argument to :obj:`True`, set the ``step`` argument to :obj:`None`.

        Returns:
            A suggested float value.

        .. seealso::
            :ref:`configurations` tutorial describes more details and flexible usages.
        """

        distribution = FloatDistribution(low, high, log=log, step=step)
        suggested_value = self._suggest(name, distribution)
        self._check_distribution(name, distribution)
        return suggested_value

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg.format(args=""))
    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value for the continuous parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`
        in the linear domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of
        :math:`\\mathsf{low}` will be returned.

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.

        Returns:
            A suggested float value.
        """

        return self.suggest_float(name, low, high)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg.format(args="(..., log=True)"))
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value for the continuous parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`
        in the log domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of
        :math:`\\mathsf{low}` will be returned.

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.

        Returns:
            A suggested float value.
        """

        return self.suggest_float(name, low, high, log=True)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg.format(args="(..., step=...)"))
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        """Suggest a value for the discrete parameter.

        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high}]`,
        and the step of discretization is :math:`q`. More specifically,
        this method returns one of the values in the sequence
        :math:`\\mathsf{low}, \\mathsf{low} + q, \\mathsf{low} + 2 q, \\dots,
        \\mathsf{low} + k q \\le \\mathsf{high}`,
        where :math:`k` denotes an integer. Note that :math:`high` may be changed due to round-off
        errors if :math:`q` is not an integer. Please check warning messages to find the changed
        values.

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
            q:
                A step of discretization.

        Returns:
            A suggested float value.
        """

        return self.suggest_float(name, low, high, step=q)

    @convert_positional_args(previous_positional_arg_names=_SUGGEST_INT_POSITIONAL_ARGS)
    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int:
        """Suggest a value for the integer parameter.

        The value is sampled from the integers in :math:`[\\mathsf{low}, \\mathsf{high}]`.

        Example:

            Suggest the number of trees in `RandomForestClassifier <https://scikit-learn.org/
            stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)


                def objective(trial):
                    n_estimators = trial.suggest_int("n_estimators", 50, 400)
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)


                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=3)

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
                ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,
                ``low`` must be larger than 0.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
                ``high`` must be greater than or equal to ``low``.
            step:
                A step of discretization.

                .. note::
                    Note that :math:`\\mathsf{high}` is modified if the range is not divisible by
                    :math:`\\mathsf{step}`. Please check the warning messages to find the changed
                    values.

                .. note::
                    The method returns one of the values in the sequence
                    :math:`\\mathsf{low}, \\mathsf{low} + \\mathsf{step}, \\mathsf{low} + 2 *
                    \\mathsf{step}, \\dots, \\mathsf{low} + k * \\mathsf{step} \\le
                    \\mathsf{high}`, where :math:`k` denotes an integer.

                .. note::
                    The ``step != 1`` and ``log`` arguments cannot be used at the same time.
                    To set the ``step`` argument :math:`\\mathsf{step} \\ge 2`, set the
                    ``log`` argument to :obj:`False`.
            log:
                A flag to sample the value from the log domain or not.

                .. note::
                    If ``log`` is true, at first, the range of suggested values is divided into
                    grid points of width 1. The range of suggested values is then converted to
                    a log domain, from which a value is sampled. The uniformly sampled
                    value is re-converted to the original domain and rounded to the nearest grid
                    point that we just split, and the suggested value is determined.
                    For example, if `low = 2` and `high = 8`, then the range of suggested values is
                    `[2, 3, 4, 5, 6, 7, 8]` and lower values tend to be more sampled than higher
                    values.

                .. note::
                    The ``step != 1`` and ``log`` arguments cannot be used at the same time.
                    To set the ``log`` argument to :obj:`True`, set the ``step`` argument to 1.

        .. seealso::
            :ref:`configurations` tutorial describes more details and flexible usages.
        """

        distribution = IntDistribution(low=low, high=high, log=log, step=step)
        suggested_value = int(self._suggest(name, distribution))
        self._check_distribution(name, distribution)
        return suggested_value

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[None]) -> None: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[bool]) -> bool: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[int]) -> int: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[float]) -> float: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str: ...

    @overload
    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType: ...

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        """Suggest a value for the categorical parameter.

        The value is sampled from ``choices``.

        Example:

            Suggest a kernel function of `SVC <https://scikit-learn.org/stable/modules/generated/
            sklearn.svm.SVC.html>`__.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.svm import SVC

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)


                def objective(trial):
                    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
                    clf = SVC(kernel=kernel, gamma="scale", random_state=0)
                    clf.fit(X_train, y_train)
                    return clf.score(X_valid, y_valid)


                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=3)


        Args:
            name:
                A parameter name.
            choices:
                Parameter value candidates.

        .. seealso::
            :class:`~optuna.distributions.CategoricalDistribution`.

        Returns:
            A suggested value.

        .. seealso::
            :ref:`configurations` tutorial describes more details and flexible usages.
        """
        # There is no need to call self._check_distribution because
        # CategoricalDistribution does not support dynamic value space.

        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        """Report an objective function value for a given step.

        The reported values are used by the pruners to determine whether this trial should be
        pruned.

        .. seealso::
            Please refer to :class:`~optuna.pruners.BasePruner`.

        .. note::
            The reported value is converted to ``float`` type by applying ``float()``
            function internally. Thus, it accepts all float-like types (e.g., ``numpy.float32``).
            If the conversion fails, a ``TypeError`` is raised.

        .. note::
            If this method is called multiple times at the same ``step`` in a trial,
            the reported ``value`` only the first time is stored and the reported values
            from the second time are ignored.

        .. note::
            :func:`~optuna.trial.Trial.report` does not support multi-objective
            optimization.

        Example:

            Report intermediate scores of `SGDClassifier <https://scikit-learn.org/stable/modules/
            generated/sklearn.linear_model.SGDClassifier.html>`__ training.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.linear_model import SGDClassifier
                from sklearn.model_selection import train_test_split

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y)


                def objective(trial):
                    clf = SGDClassifier(random_state=0)
                    for step in range(100):
                        clf.partial_fit(X_train, y_train, np.unique(y))
                        intermediate_value = clf.score(X_valid, y_valid)
                        trial.report(intermediate_value, step=step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    return clf.score(X_valid, y_valid)


                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=3)


        Args:
            value:
                A value returned from the objective function.
            step:
                Step of the trial (e.g., Epoch of neural network training). Note that pruners
                assume that ``step`` starts at zero. For example,
                :class:`~optuna.pruners.MedianPruner` simply checks if ``step`` is less than
                ``n_warmup_steps`` as the warmup mechanism.
                ``step`` must be a positive integer.
        """

        if len(self.study.directions) > 1:
            raise NotImplementedError(
                "Trial.report is not supported for multi-objective optimization."
            )

        try:
            # For convenience, we allow users to report a value that can be cast to `float`.
            value = float(value)
        except (TypeError, ValueError):
            message = (
                f"The `value` argument is of type '{type(value)}' but supposed to be a float."
            )
            raise TypeError(message) from None

        try:
            step = int(step)
        except (TypeError, ValueError):
            message = f"The `step` argument is of type '{type(step)}' but supposed to be an int."
            raise TypeError(message) from None

        if step < 0:
            raise ValueError(f"The `step` argument is {step} but cannot be negative.")

        if step in self._cached_frozen_trial.intermediate_values:
            # Do nothing if already reported.
            warnings.warn(
                f"The reported value is ignored because this `step` {step} is already reported."
            )
            return

        self.storage.set_trial_intermediate_value(self._trial_id, step, value)
        self._cached_frozen_trial.intermediate_values[step] = value

    def should_prune(self) -> bool:
        """Suggest whether the trial should be pruned or not.

        The suggestion is made by a pruning algorithm associated with the trial and is based on
        previously reported values. The algorithm can be specified when constructing a
        :class:`~optuna.study.Study`.

        .. note::
            If no values have been reported, the algorithm cannot make meaningful suggestions.
            Similarly, if this method is called multiple times with the exact same set of reported
            values, the suggestions will be the same.

        .. seealso::
            Please refer to the example code in :func:`optuna.trial.Trial.report`.

        .. note::
            :func:`~optuna.trial.Trial.should_prune` does not support multi-objective
            optimization.

        Returns:
            A boolean value. If :obj:`True`, the trial should be pruned according to the
            configured pruning algorithm. Otherwise, the trial should continue.
        """

        if len(self.study.directions) > 1:
            raise NotImplementedError(
                "Trial.should_prune is not supported for multi-objective optimization."
            )

        trial = self._get_latest_trial()
        return self.study.pruner.prune(self.study, trial)

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set user attributes to the trial.

        The user attributes in the trial can be access via :func:`optuna.trial.Trial.user_attrs`.

        .. seealso::

            See the recipe on :ref:`attributes`.

        Example:

            Save fixed hyperparameters of neural network training.

            .. testcode::

                import numpy as np
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.neural_network import MLPClassifier

                import optuna

                X, y = load_iris(return_X_y=True)
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)


                def objective(trial):
                    trial.set_user_attr("BATCHSIZE", 128)
                    momentum = trial.suggest_float("momentum", 0, 1.0)
                    clf = MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        batch_size=trial.user_attrs["BATCHSIZE"],
                        momentum=momentum,
                        solver="sgd",
                        random_state=0,
                    )
                    clf.fit(X_train, y_train)

                    return clf.score(X_valid, y_valid)


                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=3)
                assert "BATCHSIZE" in study.best_trial.user_attrs.keys()
                assert study.best_trial.user_attrs["BATCHSIZE"] == 128


        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.set_trial_user_attr(self._trial_id, key, value)
        self._cached_frozen_trial.user_attrs[key] = value

    @deprecated_func("3.1.0", "5.0.0")
    def set_system_attr(self, key: str, value: Any) -> None:
        """Set system attributes to the trial.

        Note that Optuna internally uses this method to save system messages such as failure
        reason of trials. Please use :func:`~optuna.trial.Trial.set_user_attr` to set users'
        attributes.

        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.set_trial_system_attr(self._trial_id, key, value)
        self._cached_frozen_trial.system_attrs[key] = value

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        storage = self.storage
        trial_id = self._trial_id

        trial = self._get_latest_trial()

        if name in trial.distributions:
            # No need to sample if already suggested.
            distributions.check_distribution_compatibility(trial.distributions[name], distribution)
            param_value = trial.params[name]
        else:
            if self._is_fixed_param(name, distribution):
                param_value = self._fixed_params[name]
            elif distribution.single():
                param_value = distributions._get_single_value(distribution)
            elif self._is_relative_param(name, distribution):
                param_value = self.relative_params[name]
            else:
                study = pruners._filter_study(self.study, trial)
                param_value = self.study.sampler.sample_independent(
                    study, trial, name, distribution
                )

            # `param_value` is validated here (invalid value like `np.nan` raises ValueError).
            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
            storage.set_trial_param(trial_id, name, param_value_in_internal_repr, distribution)

            self._cached_frozen_trial.distributions[name] = distribution
            self._cached_frozen_trial.params[name] = param_value
        return param_value

    def _is_fixed_param(self, name: str, distribution: BaseDistribution) -> bool:
        if name not in self._fixed_params:
            return False

        param_value = self._fixed_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)

        contained = distribution._contains(param_value_in_internal_repr)
        if not contained:
            warnings.warn(
                "Fixed parameter '{}' with value {} is out of range "
                "for distribution {}.".format(name, param_value, distribution)
            )
        return True

    def _is_relative_param(self, name: str, distribution: BaseDistribution) -> bool:
        if name not in self.relative_params:
            return False

        if name not in self.relative_search_space:
            raise ValueError(
                "The parameter '{}' was sampled by `sample_relative` method "
                "but it is not contained in the relative search space.".format(name)
            )

        relative_distribution = self.relative_search_space[name]
        distributions.check_distribution_compatibility(relative_distribution, distribution)

        param_value = self.relative_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        return distribution._contains(param_value_in_internal_repr)

    def _check_distribution(self, name: str, distribution: BaseDistribution) -> None:
        old_distribution = self._cached_frozen_trial.distributions.get(name, distribution)
        if old_distribution != distribution:
            warnings.warn(
                'Inconsistent parameter values for distribution with name "{}"! '
                "This might be a configuration mistake. "
                "Optuna allows to call the same distribution with the same "
                "name more than once in a trial. "
                "When the parameter values are inconsistent optuna only "
                "uses the values of the first call and ignores all following. "
                "Using these values: {}".format(name, old_distribution._asdict()),
                RuntimeWarning,
            )

    def _get_latest_trial(self) -> FrozenTrial:
        # TODO(eukaryo): Remove this method after `system_attrs` property is removed.
        latest_trial = copy.copy(self._cached_frozen_trial)
        latest_trial.system_attrs = _LazyTrialSystemAttrs(  # type: ignore[assignment]
            self._trial_id, self.storage
        )
        return latest_trial

    @property
    def params(self) -> dict[str, Any]:
        """Return parameters to be optimized.

        Returns:
            A dictionary containing all parameters.
        """

        return copy.deepcopy(self._cached_frozen_trial.params)

    @property
    def distributions(self) -> dict[str, BaseDistribution]:
        """Return distributions of parameters to be optimized.

        Returns:
            A dictionary containing all distributions.
        """

        return copy.deepcopy(self._cached_frozen_trial.distributions)

    @property
    def user_attrs(self) -> dict[str, Any]:
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return copy.deepcopy(self._cached_frozen_trial.user_attrs)

    @property
    @deprecated_func("3.1.0", "5.0.0")
    def system_attrs(self) -> dict[str, Any]:
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return copy.deepcopy(self.storage.get_trial_system_attrs(self._trial_id))

    @property
    def datetime_start(self) -> datetime.datetime | None:
        """Return start datetime.

        Returns:
            Datetime where the :class:`~optuna.trial.Trial` started.
        """
        return self._cached_frozen_trial.datetime_start

    @property
    def number(self) -> int:
        """Return trial's number which is consecutive and unique in a study.

        Returns:
            A trial number.
        """

        return self._cached_frozen_trial.number


class _LazyTrialSystemAttrs(UserDict):
    def __init__(self, trial_id: int, storage: optuna.storages.BaseStorage) -> None:
        super().__init__()
        self._trial_id = trial_id
        self._storage = storage
        self._initialized = False

    def __getattribute__(self, key: str) -> Any:
        if key == "data":
            if not self._initialized:
                self._initialized = True
                super().update(self._storage.get_trial_system_attrs(self._trial_id))
        return super().__getattribute__(key)
