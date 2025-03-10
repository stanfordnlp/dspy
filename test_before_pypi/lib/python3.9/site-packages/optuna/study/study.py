from __future__ import annotations

from collections.abc import Container
from collections.abc import Iterable
from collections.abc import Mapping
import copy
from numbers import Real
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union
import warnings

import numpy as np

import optuna
from optuna import exceptions
from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna._convert_positional_args import convert_positional_args
from optuna._deprecated import deprecated_func
from optuna._experimental import experimental_func
from optuna._imports import _LazyImport
from optuna._typing import JSONSerializable
from optuna.distributions import _convert_old_distribution_to_new_distribution
from optuna.distributions import BaseDistribution
from optuna.storages._heartbeat import is_heartbeat_enabled
from optuna.study._constrained_optimization import _CONSTRAINTS_KEY
from optuna.study._constrained_optimization import _get_feasible_trials
from optuna.study._multi_objective import _get_pareto_front_trials
from optuna.study._optimize import _optimize
from optuna.study._study_direction import StudyDirection
from optuna.study._study_summary import StudySummary  # NOQA
from optuna.study._tell import _tell_with_warning
from optuna.trial import create_trial
from optuna.trial import TrialState


_dataframe = _LazyImport("optuna.study._dataframe")

if TYPE_CHECKING:
    from optuna.study._dataframe import pd
    from optuna.trial import FrozenTrial
    from optuna.trial import Trial


ObjectiveFuncType = Callable[["Trial"], Union[float, Sequence[float]]]


_SYSTEM_ATTR_METRIC_NAMES = "study:metric_names"


_logger = logging.get_logger(__name__)


class _ThreadLocalStudyAttribute(threading.local):
    in_optimize_loop: bool = False
    cached_all_trials: list[FrozenTrial] | None = None


class Study:
    """A study corresponds to an optimization task, i.e., a set of trials.

    This object provides interfaces to run a new :class:`~optuna.trial.Trial`, access trials'
    history, set/get user-defined attributes of the study itself.

    Note that the direct use of this constructor is not recommended.
    To create and load a study, please refer to the documentation of
    :func:`~optuna.study.create_study` and :func:`~optuna.study.load_study` respectively.

    """

    def __init__(
        self,
        study_name: str,
        storage: str | storages.BaseStorage,
        sampler: "samplers.BaseSampler" | None = None,
        pruner: pruners.BasePruner | None = None,
    ) -> None:
        self.study_name = study_name
        storage = storages.get_storage(storage)
        study_id = storage.get_study_id_from_name(study_name)
        self._study_id = study_id
        self._storage = storage
        self._directions = storage.get_study_directions(study_id)

        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self._thread_local = _ThreadLocalStudyAttribute()
        self._stop_flag = False

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_thread_local"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._thread_local = _ThreadLocalStudyAttribute()

    @property
    def best_params(self) -> dict[str, Any]:
        """Return parameters of the best trial in the study.

        .. note::
            This feature can only be used for single-objective optimization.

        Returns:
            A dictionary containing parameters of the best trial.

        """

        return self.best_trial.params

    @property
    def best_value(self) -> float:
        """Return the best objective value in the study.

        .. note::
            This feature can only be used for single-objective optimization.

        Returns:
            A float representing the best objective value.

        """

        best_value = self.best_trial.value
        assert best_value is not None

        return best_value

    @property
    def best_trial(self) -> FrozenTrial:
        """Return the best trial in the study.

        .. note::
            This feature can only be used for single-objective optimization.
            If your study is multi-objective,
            use :attr:`~optuna.study.Study.best_trials` instead.

        Returns:
            A :class:`~optuna.trial.FrozenTrial` object of the best trial.

        .. seealso::
            The :ref:`reuse_best_trial` tutorial provides a detailed example of how to use this
            method.

        """

        if self._is_multi_objective():
            raise RuntimeError(
                "A single best trial cannot be retrieved from a multi-objective study. Consider "
                "using Study.best_trials to retrieve a list containing the best trials."
            )

        best_trial = self._storage.get_best_trial(self._study_id)

        # If the trial with the best value is infeasible, select the best trial from all feasible
        # trials. Note that the behavior is undefined when constrained optimization without the
        # violation value in the best-valued trial.
        constraints = best_trial.system_attrs.get(_CONSTRAINTS_KEY)
        if constraints is not None and any([x > 0.0 for x in constraints]):
            complete_trials = self.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            feasible_trials = _get_feasible_trials(complete_trials)
            if len(feasible_trials) == 0:
                raise ValueError("No feasible trials are completed yet.")
            if self.direction == StudyDirection.MAXIMIZE:
                best_trial = max(feasible_trials, key=lambda t: cast(float, t.value))
            else:
                best_trial = min(feasible_trials, key=lambda t: cast(float, t.value))

        return copy.deepcopy(best_trial)

    @property
    def best_trials(self) -> list[FrozenTrial]:
        """Return trials located at the Pareto front in the study.

        A trial is located at the Pareto front if there are no trials that dominate the trial.
        It's called that a trial ``t0`` dominates another trial ``t1`` if
        ``all(v0 <= v1) for v0, v1 in zip(t0.values, t1.values)`` and
        ``any(v0 < v1) for v0, v1 in zip(t0.values, t1.values)`` are held.

        Returns:
            A list of :class:`~optuna.trial.FrozenTrial` objects.
        """

        # Check whether the study is constrained optimization.
        trials = self.get_trials(deepcopy=False)
        is_constrained = any((_CONSTRAINTS_KEY in trial.system_attrs) for trial in trials)

        return _get_pareto_front_trials(self, consider_constraint=is_constrained)

    @property
    def direction(self) -> StudyDirection:
        """Return the direction of the study.

        .. note::
            This feature can only be used for single-objective optimization.
            If your study is multi-objective,
            use :attr:`~optuna.study.Study.directions` instead.

        Returns:
            A :class:`~optuna.study.StudyDirection` object.

        """

        if self._is_multi_objective():
            raise RuntimeError(
                "A single direction cannot be retrieved from a multi-objective study. Consider "
                "using Study.directions to retrieve a list containing all directions."
            )

        return self.directions[0]

    @property
    def directions(self) -> list[StudyDirection]:
        """Return the directions of the study.

        Returns:
            A list of :class:`~optuna.study.StudyDirection` objects.
        """

        return self._directions

    @property
    def trials(self) -> list[FrozenTrial]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        This is a short form of ``self.get_trials(deepcopy=True, states=None)``.

        Returns:
            A list of :class:`~optuna.trial.FrozenTrial` objects.

            .. seealso::
                See :func:`~optuna.study.Study.get_trials` for related method.

        """

        return self.get_trials(deepcopy=True, states=None)

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        .. seealso::
            See :attr:`~optuna.study.Study.trials` for related property.

        Example:
            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", -1, 1)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                trials = study.get_trials()
                assert len(trials) == 3
        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.
                Note that if you set the flag to :obj:`False`, you shouldn't mutate
                any fields of the returned trial. Otherwise the internal state of
                the study may corrupt and unexpected behavior may happen.
            states:
                Trial states to filter on. If :obj:`None`, include all states.

        Returns:
            A list of :class:`~optuna.trial.FrozenTrial` objects.
        """
        return self._get_trials(deepcopy, states, use_cache=False)

    def _get_trials(
        self,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
        use_cache: bool = False,
    ) -> list[FrozenTrial]:
        if use_cache:
            if self._thread_local.cached_all_trials is None:
                self._thread_local.cached_all_trials = self._storage.get_all_trials(
                    self._study_id, deepcopy=False
                )
            trials = self._thread_local.cached_all_trials
            if states is not None:
                filtered_trials = [t for t in trials if t.state in states]
            else:
                filtered_trials = trials
            return copy.deepcopy(filtered_trials) if deepcopy else filtered_trials

        return self._storage.get_all_trials(self._study_id, deepcopy=deepcopy, states=states)

    @property
    def user_attrs(self) -> dict[str, Any]:
        """Return user attributes.

        .. seealso::

            See :func:`~optuna.study.Study.set_user_attr` for related method.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", 0, 1)
                    y = trial.suggest_float("y", 0, 1)
                    return x**2 + y**2


                study = optuna.create_study()

                study.set_user_attr("objective function", "quadratic function")
                study.set_user_attr("dimensions", 2)
                study.set_user_attr("contributors", ["Akiba", "Sano"])

                assert study.user_attrs == {
                    "objective function": "quadratic function",
                    "dimensions": 2,
                    "contributors": ["Akiba", "Sano"],
                }

        Returns:
            A dictionary containing all user attributes.
        """

        return copy.deepcopy(self._storage.get_study_user_attrs(self._study_id))

    @property
    @deprecated_func("3.1.0", "5.0.0")
    def system_attrs(self) -> dict[str, Any]:
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return copy.deepcopy(self._storage.get_study_system_attrs(self._study_id))

    @property
    def metric_names(self) -> list[str] | None:
        """Return metric names.

        .. note::
            Use :meth:`~optuna.study.Study.set_metric_names` to set the metric names first.

        Returns:
            A list with names for each dimension of the returned values of the objective function.
        """
        return self._storage.get_study_system_attrs(self._study_id).get(_SYSTEM_ATTR_METRIC_NAMES)

    def optimize(
        self,
        func: ObjectiveFuncType,
        n_trials: int | None = None,
        timeout: float | None = None,
        n_jobs: int = 1,
        catch: Iterable[type[Exception]] | type[Exception] = (),
        callbacks: Iterable[Callable[[Study, FrozenTrial], None]] | None = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        """Optimize an objective function.

        Optimization is done by choosing a suitable set of hyperparameter values from a given
        range. Uses a sampler which implements the task of value suggestion based on a specified
        distribution. The sampler is specified in :func:`~optuna.study.create_study` and the
        default choice for the sampler is TPE.
        See also :class:`~optuna.samplers.TPESampler` for more details on 'TPE'.

        Optimization will be stopped when receiving a termination signal such as SIGINT and
        SIGTERM. Unlike other signals, a trial is automatically and cleanly failed when receiving
        SIGINT (Ctrl+C). If ``n_jobs`` is greater than one or if another signal than SIGINT
        is used, the interrupted trial state won't be properly updated.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", -1, 1)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

        Args:
            func:
                A callable that implements objective function.
            n_trials:
                The number of trials for each process. :obj:`None` represents no limit in terms of
                the number of trials. The study continues to create trials until the number of
                trials reaches ``n_trials``, ``timeout`` period elapses,
                :func:`~optuna.study.Study.stop` is called, or a termination signal such as
                SIGTERM or Ctrl+C is received.

                .. seealso::
                    :class:`optuna.study.MaxTrialsCallback` can ensure how many times trials
                    will be performed across all processes.
            timeout:
                Stop study after the given number of second(s). :obj:`None` represents no limit in
                terms of elapsed time. The study continues to create trials until the number of
                trials reaches ``n_trials``, ``timeout`` period elapses,
                :func:`~optuna.study.Study.stop` is called or, a termination signal such as
                SIGTERM or Ctrl+C is received.
            n_jobs:
                The number of parallel jobs. If this argument is set to ``-1``, the number is
                set to CPU count.

                .. note::
                    ``n_jobs`` allows parallelization using :obj:`threading` and may suffer from
                    `Python's GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`__.
                    It is recommended to use :ref:`process-based parallelization<distributed>`
                    if ``func`` is CPU bound.

            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~optuna.exceptions.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
                :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.

                .. seealso::

                    See the tutorial of :ref:`optuna_callback` for how to use and implement
                    callback functions.

            gc_after_trial:
                Flag to determine whether to automatically run garbage collection after each trial.
                Set to :obj:`True` to run the garbage collection, :obj:`False` otherwise.
                When it runs, it runs a full collection by internally calling :func:`gc.collect`.
                If you see an increase in memory consumption over several trials, try setting this
                flag to :obj:`True`.

                .. seealso::

                    :ref:`out-of-memory-gc-collect`

            show_progress_bar:
                Flag to show progress bars or not. To show progress bar, set this :obj:`True`.
                Note that it is disabled when ``n_trials`` is :obj:`None`,
                ``timeout`` is not :obj:`None`, and ``n_jobs`` :math:`\\ne 1`.

        Raises:
            RuntimeError:
                If nested invocation of this method occurs.
        """
        _optimize(
            study=self,
            func=func,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=tuple(catch) if isinstance(catch, Iterable) else (catch,),
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

    def ask(self, fixed_distributions: dict[str, BaseDistribution] | None = None) -> Trial:
        """Create a new trial from which hyperparameters can be suggested.

        This method is part of an alternative to :func:`~optuna.study.Study.optimize` that allows
        controlling the lifetime of a trial outside the scope of ``func``. Each call to this
        method should be followed by a call to :func:`~optuna.study.Study.tell` to finish the
        created trial.

        .. seealso::

            The :ref:`ask_and_tell` tutorial provides use-cases with examples.

        Example:

            Getting the trial object with the :func:`~optuna.study.Study.ask` method.

            .. testcode::

                import optuna


                study = optuna.create_study()

                trial = study.ask()

                x = trial.suggest_float("x", -1, 1)

                study.tell(trial, x**2)

        Example:

            Passing previously defined distributions to the :func:`~optuna.study.Study.ask`
            method.

            .. testcode::

                import optuna


                study = optuna.create_study()

                distributions = {
                    "optimizer": optuna.distributions.CategoricalDistribution(["adam", "sgd"]),
                    "lr": optuna.distributions.FloatDistribution(0.0001, 0.1, log=True),
                }

                # You can pass the distributions previously defined.
                trial = study.ask(fixed_distributions=distributions)

                # `optimizer` and `lr` are already suggested and accessible with `trial.params`.
                assert "optimizer" in trial.params
                assert "lr" in trial.params

        Args:
            fixed_distributions:
                A dictionary containing the parameter names and parameter's distributions. Each
                parameter in this dictionary is automatically suggested for the returned trial,
                even when the suggest method is not explicitly invoked by the user. If this
                argument is set to :obj:`None`, no parameter is automatically suggested.

        Returns:
            A :class:`~optuna.trial.Trial`.
        """

        if not self._thread_local.in_optimize_loop and is_heartbeat_enabled(self._storage):
            warnings.warn("Heartbeat of storage is supposed to be used with Study.optimize.")

        fixed_distributions = fixed_distributions or {}
        fixed_distributions = {
            key: _convert_old_distribution_to_new_distribution(dist)
            for key, dist in fixed_distributions.items()
        }

        # Sync storage once every trial.
        self._thread_local.cached_all_trials = None

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = optuna.Trial(self, trial_id)

        for name, param in fixed_distributions.items():
            trial._suggest(name, param)

        return trial

    def tell(
        self,
        trial: Trial | int,
        values: float | Sequence[float] | None = None,
        state: TrialState | None = None,
        skip_if_finished: bool = False,
    ) -> FrozenTrial:
        """Finish a trial created with :func:`~optuna.study.Study.ask`.

        .. seealso::

            The :ref:`ask_and_tell` tutorial provides use-cases with examples.

        Example:

            .. testcode::

                import optuna
                from optuna.trial import TrialState


                def f(x):
                    return (x - 2) ** 2


                def df(x):
                    return 2 * x - 4


                study = optuna.create_study()

                n_trials = 30

                for _ in range(n_trials):
                    trial = study.ask()

                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

                    # Iterative gradient descent objective function.
                    x = 3  # Initial value.
                    for step in range(128):
                        y = f(x)

                        trial.report(y, step=step)

                        if trial.should_prune():
                            # Finish the trial with the pruned state.
                            study.tell(trial, state=TrialState.PRUNED)
                            break

                        gy = df(x)
                        x -= gy * lr
                    else:
                        # Finish the trial with the final value after all iterations.
                        study.tell(trial, y)

        Args:
            trial:
                A :class:`~optuna.trial.Trial` object or a trial number.
            values:
                Optional objective value or a sequence of such values in case the study is used
                for multi-objective optimization. Argument must be provided if ``state`` is
                :class:`~optuna.trial.TrialState.COMPLETE` and should be :obj:`None` if ``state``
                is :class:`~optuna.trial.TrialState.FAIL` or
                :class:`~optuna.trial.TrialState.PRUNED`.
            state:
                State to be reported. Must be :obj:`None`,
                :class:`~optuna.trial.TrialState.COMPLETE`,
                :class:`~optuna.trial.TrialState.FAIL` or
                :class:`~optuna.trial.TrialState.PRUNED`.
                If ``state`` is :obj:`None`,
                it will be updated to :class:`~optuna.trial.TrialState.COMPLETE`
                or :class:`~optuna.trial.TrialState.FAIL` depending on whether
                validation for ``values`` reported succeed or not.
            skip_if_finished:
                Flag to control whether exception should be raised when values for already
                finished trial are told. If :obj:`True`, tell is skipped without any error
                when the trial is already finished.

        Returns:
            A :class:`~optuna.trial.FrozenTrial` representing the resulting trial.
            A returned trial is deep copied thus user can modify it as needed.
        """

        return _tell_with_warning(
            study=self,
            trial=trial,
            value_or_values=values,
            state=state,
            skip_if_finished=skip_if_finished,
        )

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a user attribute to the study.

        .. seealso::

            See :attr:`~optuna.study.Study.user_attrs` for related attribute.

        .. seealso::

            See the recipe on :ref:`attributes`.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", 0, 1)
                    y = trial.suggest_float("y", 0, 1)
                    return x**2 + y**2


                study = optuna.create_study()

                study.set_user_attr("objective function", "quadratic function")
                study.set_user_attr("dimensions", 2)
                study.set_user_attr("contributors", ["Akiba", "Sano"])

                assert study.user_attrs == {
                    "objective function": "quadratic function",
                    "dimensions": 2,
                    "contributors": ["Akiba", "Sano"],
                }

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_user_attr(self._study_id, key, value)

    @deprecated_func("3.1.0", "5.0.0")
    def set_system_attr(self, key: str, value: Any) -> None:
        """Set a system attribute to the study.

        Note that Optuna internally uses this method to save system messages. Please use
        :func:`~optuna.study.Study.set_user_attr` to set users' attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_system_attr(self._study_id, key, value)

    def trials_dataframe(
        self,
        attrs: tuple[str, ...] = (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        ),
        multi_index: bool = False,
    ) -> "pd.DataFrame":
        """Export trials as a pandas DataFrame_.

        The DataFrame_ provides various features to analyze studies. It is also useful to draw a
        histogram of objective values and to export trials as a CSV file.
        If there are no trials, an empty DataFrame_ is returned.

        Example:

            .. testcode::

                import optuna
                import pandas


                def objective(trial):
                    x = trial.suggest_float("x", -1, 1)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                # Create a dataframe from the study.
                df = study.trials_dataframe()
                assert isinstance(df, pandas.DataFrame)
                assert df.shape[0] == 3  # n_trials.

        Args:
            attrs:
                Specifies field names of :class:`~optuna.trial.FrozenTrial` to include them to a
                DataFrame of trials.
            multi_index:
                Specifies whether the returned DataFrame_ employs MultiIndex_ or not. Columns that
                are hierarchical by nature such as ``(params, x)`` will be flattened to
                ``params_x`` when set to :obj:`False`.

        Returns:
            A pandas DataFrame_ of trials in the :class:`~optuna.study.Study`.

        .. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
        .. _MultiIndex: https://pandas.pydata.org/pandas-docs/stable/advanced.html

        Note:
            If ``value`` is in ``attrs`` during multi-objective optimization, it is implicitly
            replaced with ``values``.

        Note:
            If :meth:`~optuna.study.Study.set_metric_names` is called, the ``value`` or ``values``
            is implicitly replaced with the dictionary with the objective name as key and the
            objective value as value.
        """
        return _dataframe._trials_dataframe(self, attrs, multi_index)

    def stop(self) -> None:
        """Exit from the current optimization loop after the running trials finish.

        This method lets the running :meth:`~optuna.study.Study.optimize` method return
        immediately after all trials which the :meth:`~optuna.study.Study.optimize` method
        spawned finishes.
        This method does not affect any behaviors of parallel or successive study processes.
        This method only works when it is called inside an objective function or callback.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    if trial.number == 4:
                        trial.study.stop()
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=10)
                assert len(study.trials) == 5

        """

        if not self._thread_local.in_optimize_loop:
            raise RuntimeError(
                "`Study.stop` is supposed to be invoked inside an objective function or a "
                "callback."
            )

        self._stop_flag = True

    def enqueue_trial(
        self,
        params: dict[str, Any],
        user_attrs: dict[str, Any] | None = None,
        skip_if_exists: bool = False,
    ) -> None:
        """Enqueue a trial with given parameter values.

        You can fix the next sampling parameters which will be evaluated in your
        objective function.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


                study = optuna.create_study()
                study.enqueue_trial({"x": 5})
                study.enqueue_trial({"x": 0}, user_attrs={"memo": "optimal"})
                study.optimize(objective, n_trials=2)

                assert study.trials[0].params == {"x": 5}
                assert study.trials[1].params == {"x": 0}
                assert study.trials[1].user_attrs == {"memo": "optimal"}

        Args:
            params:
                Parameter values to pass your objective function.
            user_attrs:
                A dictionary of user-specific attributes other than ``params``.
            skip_if_exists:
                When :obj:`True`, prevents duplicate trials from being enqueued again.

                .. note::
                    This method might produce duplicated trials if called simultaneously
                    by multiple processes at the same time with same ``params`` dict.

        .. seealso::

            Please refer to :ref:`enqueue_trial_tutorial` for the tutorial of specifying
            hyperparameters manually.
        """

        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary.")

        if skip_if_exists and self._should_skip_enqueue(params):
            _logger.info(f"Trial with params {params} already exists. Skipping enqueue.")
            return

        self.add_trial(
            create_trial(
                state=TrialState.WAITING,
                system_attrs={"fixed_params": params},
                user_attrs=user_attrs,
            )
        )

    def add_trial(self, trial: FrozenTrial) -> None:
        """Add trial to study.

        The trial is validated before being added.

        Example:

            .. testcode::

                import optuna
                from optuna.distributions import FloatDistribution


                def objective(trial):
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


                study = optuna.create_study()
                assert len(study.trials) == 0

                trial = optuna.trial.create_trial(
                    params={"x": 2.0},
                    distributions={"x": FloatDistribution(0, 10)},
                    value=4.0,
                )

                study.add_trial(trial)
                assert len(study.trials) == 1

                study.optimize(objective, n_trials=3)
                assert len(study.trials) == 4

                other_study = optuna.create_study()

                for trial in study.trials:
                    other_study.add_trial(trial)
                assert len(other_study.trials) == len(study.trials)

                other_study.optimize(objective, n_trials=2)
                assert len(other_study.trials) == len(study.trials) + 2

        .. seealso::

            This method should in general be used to add already evaluated trials
            (``trial.state.is_finished() == True``). To queue trials for evaluation,
            please refer to :func:`~optuna.study.Study.enqueue_trial`.

        .. seealso::

            See :func:`~optuna.trial.create_trial` for how to create trials.

        .. seealso::
            Please refer to :ref:`add_trial_tutorial` for the tutorial of specifying
            hyperparameters with the evaluated value manually.

        Args:
            trial: Trial to add.

        """

        trial._validate()

        if trial.values is not None and len(self.directions) != len(trial.values):
            raise ValueError(
                f"The added trial has {len(trial.values)} values, which is different from the "
                f"number of objectives {len(self.directions)} in the study (determined by "
                "Study.directions)."
            )

        self._storage.create_new_trial(self._study_id, template_trial=trial)

    def add_trials(self, trials: Iterable[FrozenTrial]) -> None:
        """Add trials to study.

        The trials are validated before being added.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)
                assert len(study.trials) == 3

                other_study = optuna.create_study()
                other_study.add_trials(study.trials)
                assert len(other_study.trials) == len(study.trials)

                other_study.optimize(objective, n_trials=2)
                assert len(other_study.trials) == len(study.trials) + 2

        .. seealso::

            See :func:`~optuna.study.Study.add_trial` for addition of each trial.

        Args:
            trials: Trials to add.

        """

        for trial in trials:
            self.add_trial(trial)

    @experimental_func("3.2.0")
    def set_metric_names(self, metric_names: list[str]) -> None:
        """Set metric names.

        This method names each dimension of the returned values of the objective function.
        It is particularly useful in multi-objective optimization. The metric names are
        mainly referenced by the visualization functions.

        Example:

            .. testcode::

                import optuna
                import pandas


                def objective(trial):
                    x = trial.suggest_float("x", 0, 10)
                    return x**2, x + 1


                study = optuna.create_study(directions=["minimize", "minimize"])
                study.set_metric_names(["x**2", "x+1"])
                study.optimize(objective, n_trials=3)

                df = study.trials_dataframe(multi_index=True)
                assert isinstance(df, pandas.DataFrame)
                assert list(df.get("values").keys()) == ["x**2", "x+1"]

        .. seealso::
            The names set by this method are used in :meth:`~optuna.study.Study.trials_dataframe`
            and :func:`~optuna.visualization.plot_pareto_front`.

        Args:
            metric_names: A list of metric names for the objective function.
        """
        if len(self.directions) != len(metric_names):
            raise ValueError("The number of objectives must match the length of the metric names.")

        self._storage.set_study_system_attr(
            self._study_id, _SYSTEM_ATTR_METRIC_NAMES, metric_names
        )

    def _is_multi_objective(self) -> bool:
        """Return :obj:`True` if the study has multiple objectives.

        Returns:
            A boolean value indicates if `self.directions` has more than 1 element or not.
        """

        return len(self.directions) > 1

    def _pop_waiting_trial_id(self) -> int | None:
        for trial in self._storage.get_all_trials(
            self._study_id, deepcopy=False, states=(TrialState.WAITING,)
        ):
            if not self._storage.set_trial_state_values(trial._trial_id, state=TrialState.RUNNING):
                continue

            _logger.debug("Trial {} popped from the trial queue.".format(trial.number))
            return trial._trial_id

        return None

    def _should_skip_enqueue(self, params: Mapping[str, JSONSerializable]) -> bool:
        for trial in self.get_trials(deepcopy=False):
            trial_params = trial.system_attrs.get("fixed_params", trial.params)
            if trial_params.keys() != params.keys():
                # Can't have repeated trials if different params are suggested.
                continue

            repeated_params: list[bool] = []
            for param_name, param_value in params.items():
                existing_param = trial_params[param_name]
                if not isinstance(param_value, type(existing_param)):
                    # Enqueued param has distribution that does not match existing param
                    # (e.g. trying to enqueue categorical to float param).
                    # We are not doing anything about it here, since sanitization should
                    # be handled regardless if `skip_if_exists` is `True`.
                    repeated_params.append(False)
                    continue

                is_repeated = (
                    np.isnan(float(param_value))
                    or np.isclose(float(param_value), float(existing_param), atol=0.0)
                    if isinstance(param_value, Real)
                    else param_value == existing_param
                )
                repeated_params.append(bool(is_repeated))

            if all(repeated_params):
                return True

        return False

    def _log_completed_trial(self, trial: FrozenTrial) -> None:
        if not _logger.isEnabledFor(logging.INFO):
            return

        metric_names = self.metric_names

        if len(trial.values) > 1:
            trial_values: list[float] | dict[str, float]
            if metric_names is None:
                trial_values = trial.values
            else:
                trial_values = {name: value for name, value in zip(metric_names, trial.values)}
            _logger.info(
                "Trial {} finished with values: {} and parameters: {}.".format(
                    trial.number, trial_values, trial.params
                )
            )
        elif len(trial.values) == 1:
            trial_value: float | dict[str, float]
            if metric_names is None:
                trial_value = trial.values[0]
            else:
                trial_value = {metric_names[0]: trial.values[0]}

            message = (
                f"Trial {trial.number} finished with value: {trial_value} and parameters: "
                f"{trial.params}."
            )
            try:
                best_trial = self.best_trial
                message += f" Best is trial {best_trial.number} with value: {best_trial.value}."
            except ValueError:
                # If no feasible trials are completed yet, study.best_trial raises ValueError.
                pass
            _logger.info(message)
        else:
            assert False, "Should not reach."


@convert_positional_args(
    previous_positional_arg_names=[
        "storage",
        "sampler",
        "pruner",
        "study_name",
        "direction",
        "load_if_exists",
    ],
)
def create_study(
    *,
    storage: str | storages.BaseStorage | None = None,
    sampler: "samplers.BaseSampler" | None = None,
    pruner: pruners.BasePruner | None = None,
    study_name: str | None = None,
    direction: str | StudyDirection | None = None,
    load_if_exists: bool = False,
    directions: Sequence[str | StudyDirection] | None = None,
) -> Study:
    """Create a new :class:`~optuna.study.Study`.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 10)
                return x**2


            study = optuna.create_study()
            study.optimize(objective, n_trials=3)

    Args:
        storage:
            Database URL. If this argument is set to None,
            :class:`~optuna.storages.InMemoryStorage` is used, and the
            :class:`~optuna.study.Study` will not be persistent.

            .. note::
                When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                the database. Please refer to `SQLAlchemy's document`_ for further details.
                If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                pass it to the ``storage`` argument instead of a URL.

             .. _SQLAlchemy: https://www.sqlalchemy.org/
             .. _SQLAlchemy's document:
                 https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
             .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html

        sampler:
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used during
            single-objective optimization and :class:`~optuna.samplers.NSGAIISampler` during
            multi-objective optimization. See also :class:`~optuna.samplers`.
        pruner:
            A pruner object that decides early stopping of unpromising trials. If :obj:`None`
            is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
            also :class:`~optuna.pruners`.
        study_name:
            Study's name. If this argument is set to None, a unique name is generated
            automatically.
        direction:
            Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for
            maximization. You can also pass the corresponding :class:`~optuna.study.StudyDirection`
            object. ``direction`` and ``directions`` must not be specified at the same time.

            .. note::
                If none of `direction` and `directions` are specified, the direction of the study
                is set to "minimize".
        load_if_exists:
            Flag to control the behavior to handle a conflict of study names.
            In the case where a study named ``study_name`` already exists in the ``storage``,
            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.
        directions:
            A sequence of directions during multi-objective optimization.
            ``direction`` and ``directions`` must not be specified at the same time.

    Returns:
        A :class:`~optuna.study.Study` object.

    See also:
        :func:`optuna.create_study` is an alias of :func:`optuna.study.create_study`.

    See also:
        The :ref:`rdb` tutorial provides concrete examples to save and resume optimization using
        RDB.

    """

    if direction is None and directions is None:
        directions = ["minimize"]
    elif direction is not None and directions is not None:
        raise ValueError("Specify only one of `direction` and `directions`.")
    elif direction is not None:
        directions = [direction]
    elif directions is not None:
        directions = list(directions)
    else:
        assert False

    if len(directions) < 1:
        raise ValueError("The number of objectives must be greater than 0.")
    elif any(
        d not in ["minimize", "maximize", StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
        for d in directions
    ):
        raise ValueError(
            "Please set either 'minimize' or 'maximize' to direction. You can also set the "
            "corresponding `StudyDirection` member."
        )

    direction_objects = [
        d if isinstance(d, StudyDirection) else StudyDirection[d.upper()] for d in directions
    ]

    storage = storages.get_storage(storage)
    try:
        study_id = storage.create_new_study(direction_objects, study_name)
    except exceptions.DuplicatedStudyError:
        if load_if_exists:
            assert study_name is not None

            _logger.info(
                "Using an existing study with name '{}' instead of "
                "creating a new one.".format(study_name)
            )
            study_id = storage.get_study_id_from_name(study_name)
        else:
            raise

    if sampler is None and len(direction_objects) > 1:
        sampler = samplers.NSGAIISampler()

    study_name = storage.get_study_name_from_id(study_id)
    study = Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)

    return study


@convert_positional_args(
    previous_positional_arg_names=[
        "study_name",
        "storage",
        "sampler",
        "pruner",
    ],
)
def load_study(
    *,
    study_name: str | None,
    storage: str | storages.BaseStorage,
    sampler: "samplers.BaseSampler" | None = None,
    pruner: pruners.BasePruner | None = None,
) -> Study:
    """Load the existing :class:`~optuna.study.Study` that has the specified name.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 10)
                return x**2


            study = optuna.create_study(storage="sqlite:///example.db", study_name="my_study")
            study.optimize(objective, n_trials=3)

            loaded_study = optuna.load_study(study_name="my_study", storage="sqlite:///example.db")
            assert len(loaded_study.trials) == len(study.trials)

        .. testcleanup::

            os.remove("example.db")

    Args:
        study_name:
            Study's name. Each study has a unique name as an identifier. If :obj:`None`, checks
            whether the storage contains a single study, and if so loads that study.
            ``study_name`` is required if there are multiple studies in the storage.
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.
        sampler:
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used
            as the default. See also :class:`~optuna.samplers`.
        pruner:
            A pruner object that decides early stopping of unpromising trials.
            If :obj:`None` is specified, :class:`~optuna.pruners.MedianPruner` is used
            as the default. See also :class:`~optuna.pruners`.

    Returns:
        A :class:`~optuna.study.Study` object.

    See also:
        :func:`optuna.load_study` is an alias of :func:`optuna.study.load_study`.

    """
    if study_name is None:
        study_names = get_all_study_names(storage)
        if len(study_names) != 1:
            raise ValueError(
                f"Could not determine the study name since the storage {storage} does not "
                "contain exactly 1 study. Specify `study_name`."
            )
        study_name = study_names[0]
        _logger.info(
            f"Study name was omitted but trying to load '{study_name}' because that was the only "
            "study found in the storage."
        )

    study = Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)
    if sampler is None and len(study.directions) > 1:
        study.sampler = samplers.NSGAIISampler()
    return study


@convert_positional_args(
    previous_positional_arg_names=[
        "study_name",
        "storage",
    ],
)
def delete_study(
    *,
    study_name: str,
    storage: str | storages.BaseStorage,
) -> None:
    """Delete a :class:`~optuna.study.Study` object.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study(study_name="example-study", storage="sqlite:///example.db")
            study.optimize(objective, n_trials=3)

            optuna.delete_study(study_name="example-study", storage="sqlite:///example.db")

        .. testcleanup::

            os.remove("example.db")

    Args:
        study_name:
            Study's name.
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.

    See also:
        :func:`optuna.delete_study` is an alias of :func:`optuna.study.delete_study`.

    """

    storage = storages.get_storage(storage)
    study_id = storage.get_study_id_from_name(study_name)
    storage.delete_study(study_id)


@convert_positional_args(
    previous_positional_arg_names=[
        "from_study_name",
        "from_storage",
        "to_storage",
        "to_study_name",
    ],
    warning_stacklevel=3,
)
def copy_study(
    *,
    from_study_name: str,
    from_storage: str | storages.BaseStorage,
    to_storage: str | storages.BaseStorage,
    to_study_name: str | None = None,
) -> None:
    """Copy study from one storage to another.

    The direction(s) of the objective(s) in the study, trials, user attributes and system
    attributes are copied.

    .. note::
        :func:`~optuna.copy_study` copies a study even if the optimization is working on.
        It means users will get a copied study that contains a trial that is not finished.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")
            if os.path.exists("example_copy.db"):
                raise RuntimeError("'example_copy.db' already exists. Please remove it.")

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study(
                study_name="example-study",
                storage="sqlite:///example.db",
            )
            study.optimize(objective, n_trials=3)

            optuna.copy_study(
                from_study_name="example-study",
                from_storage="sqlite:///example.db",
                to_storage="sqlite:///example_copy.db",
            )

            study = optuna.load_study(
                study_name=None,
                storage="sqlite:///example_copy.db",
            )

        .. testcleanup::

            os.remove("example.db")
            os.remove("example_copy.db")

    Args:
        from_study_name:
            Name of study.
        from_storage:
            Source database URL such as ``sqlite:///example.db``. Please see also the
            documentation of :func:`~optuna.study.create_study` for further details.
        to_storage:
            Destination database URL.
        to_study_name:
            Name of the created study. If omitted, ``from_study_name`` is used.

    Raises:
        :class:`~optuna.exceptions.DuplicatedStudyError`:
            If a study with a conflicting name already exists in the destination storage.

    """

    from_study = load_study(study_name=from_study_name, storage=from_storage)
    to_study = create_study(
        study_name=to_study_name or from_study_name,
        storage=to_storage,
        directions=from_study.directions,
        load_if_exists=False,
    )

    for key, value in from_study._storage.get_study_system_attrs(from_study._study_id).items():
        to_study._storage.set_study_system_attr(to_study._study_id, key, value)

    for key, value in from_study.user_attrs.items():
        to_study.set_user_attr(key, value)

    # Trials are deep copied on `add_trials`.
    to_study.add_trials(from_study.get_trials(deepcopy=False))


def get_all_study_summaries(
    storage: str | storages.BaseStorage, include_best_trial: bool = True
) -> list[StudySummary]:
    """Get all history of studies stored in a specified storage.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study(study_name="example-study", storage="sqlite:///example.db")
            study.optimize(objective, n_trials=3)

            study_summaries = optuna.study.get_all_study_summaries(storage="sqlite:///example.db")
            assert len(study_summaries) == 1

            study_summary = study_summaries[0]
            assert study_summary.study_name == "example-study"

        .. testcleanup::

            os.remove("example.db")

    Args:
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.
        include_best_trial:
            Include the best trials if exist. It potentially increases the number of queries and
            may take longer to fetch summaries depending on the storage.

    Returns:
        List of study history summarized as :class:`~optuna.study.StudySummary` objects.

    See also:
        :func:`optuna.get_all_study_summaries` is an alias of
        :func:`optuna.study.get_all_study_summaries`.

    """

    storage = storages.get_storage(storage)
    frozen_studies = storage.get_all_studies()
    study_summaries = []

    for s in frozen_studies:
        all_trials = storage.get_all_trials(s._study_id)
        completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]

        n_trials = len(all_trials)

        if len(s.directions) == 1:
            direction = s.direction
            directions = None
            if include_best_trial and len(completed_trials) != 0:
                if direction == StudyDirection.MAXIMIZE:
                    best_trial = max(completed_trials, key=lambda t: cast(float, t.value))
                else:
                    best_trial = min(completed_trials, key=lambda t: cast(float, t.value))
            else:
                best_trial = None
        else:
            direction = None
            directions = s.directions
            best_trial = None

        datetime_start = min(
            [t.datetime_start for t in all_trials if t.datetime_start is not None], default=None
        )

        study_summaries.append(
            StudySummary(
                study_name=s.study_name,
                direction=direction,
                best_trial=best_trial,
                user_attrs=s.user_attrs,
                system_attrs=s.system_attrs,
                n_trials=n_trials,
                datetime_start=datetime_start,
                study_id=s._study_id,
                directions=directions,
            )
        )

    return study_summaries


def get_all_study_names(storage: str | storages.BaseStorage) -> list[str]:
    """Get all study names stored in a specified storage.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study(study_name="example-study", storage="sqlite:///example.db")
            study.optimize(objective, n_trials=3)

            study_names = optuna.study.get_all_study_names(storage="sqlite:///example.db")
            assert len(study_names) == 1

            assert study_names[0] == "example-study"

        .. testcleanup::

            os.remove("example.db")

    Args:
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.

    Returns:
        List of all study names in the storage.

    See also:
        :func:`optuna.get_all_study_names` is an alias of
        :func:`optuna.study.get_all_study_names`.

    """

    storage = storages.get_storage(storage)
    study_names = [study.study_name for study in storage.get_all_studies()]

    return study_names
