from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import datetime
import gc
import itertools
import os
import sys
from typing import Any
import warnings

import optuna
from optuna import exceptions
from optuna import logging
from optuna import progress_bar as pbar_module
from optuna import trial as trial_module
from optuna.storages._heartbeat import get_heartbeat_thread
from optuna.storages._heartbeat import is_heartbeat_enabled
from optuna.study._tell import _tell_with_warning
from optuna.study._tell import STUDY_TELL_WARNING_KEY
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)


def _optimize(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    n_trials: int | None = None,
    timeout: float | None = None,
    n_jobs: int = 1,
    catch: tuple[type[Exception], ...] = (),
    callbacks: Iterable[Callable[["optuna.Study", FrozenTrial], None]] | None = None,
    gc_after_trial: bool = False,
    show_progress_bar: bool = False,
) -> None:
    if not isinstance(catch, tuple):
        raise TypeError(
            "The catch argument is of type '{}' but must be a tuple.".format(type(catch).__name__)
        )

    if study._thread_local.in_optimize_loop:
        raise RuntimeError("Nested invocation of `Study.optimize` method isn't allowed.")

    if show_progress_bar and n_trials is None and timeout is not None and n_jobs != 1:
        warnings.warn("The timeout-based progress bar is not supported with n_jobs != 1.")
        show_progress_bar = False

    progress_bar = pbar_module._ProgressBar(show_progress_bar, n_trials, timeout)

    study._stop_flag = False

    try:
        if n_jobs == 1:
            _optimize_sequential(
                study,
                func,
                n_trials,
                timeout,
                catch,
                callbacks,
                gc_after_trial,
                reseed_sampler_rng=False,
                time_start=None,
                progress_bar=progress_bar,
            )
        else:
            if n_jobs == -1:
                n_jobs = os.cpu_count() or 1

            time_start = datetime.datetime.now()
            futures: set[Future] = set()

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                for n_submitted_trials in itertools.count():
                    if study._stop_flag:
                        break

                    if (
                        timeout is not None
                        and (datetime.datetime.now() - time_start).total_seconds() > timeout
                    ):
                        break

                    if n_trials is not None and n_submitted_trials >= n_trials:
                        break

                    if len(futures) >= n_jobs:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        # Raise if exception occurred in executing the completed futures.
                        for f in completed:
                            f.result()

                    futures.add(
                        executor.submit(
                            _optimize_sequential,
                            study,
                            func,
                            1,
                            timeout,
                            catch,
                            callbacks,
                            gc_after_trial,
                            True,
                            time_start,
                            progress_bar,
                        )
                    )
    finally:
        study._thread_local.in_optimize_loop = False
        progress_bar.close()


def _optimize_sequential(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    n_trials: int | None,
    timeout: float | None,
    catch: tuple[type[Exception], ...],
    callbacks: Iterable[Callable[["optuna.Study", FrozenTrial], None]] | None,
    gc_after_trial: bool,
    reseed_sampler_rng: bool,
    time_start: datetime.datetime | None,
    progress_bar: pbar_module._ProgressBar | None,
) -> None:
    # Here we set `in_optimize_loop = True`, not at the beginning of the `_optimize()` function.
    # Because it is a thread-local object and `n_jobs` option spawns new threads.
    study._thread_local.in_optimize_loop = True
    if reseed_sampler_rng:
        study.sampler.reseed_rng()

    i_trial = 0

    if time_start is None:
        time_start = datetime.datetime.now()

    while True:
        if study._stop_flag:
            break

        if n_trials is not None:
            if i_trial >= n_trials:
                break
            i_trial += 1

        if timeout is not None:
            elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
            if elapsed_seconds >= timeout:
                break

        try:
            frozen_trial = _run_trial(study, func, catch)
        finally:
            # The following line mitigates memory problems that can be occurred in some
            # environments (e.g., services that use computing containers such as GitHub Actions).
            # Please refer to the following PR for further details:
            # https://github.com/optuna/optuna/pull/325.
            if gc_after_trial:
                gc.collect()

        if callbacks is not None:
            for callback in callbacks:
                callback(study, frozen_trial)

        if progress_bar is not None:
            elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
            progress_bar.update(elapsed_seconds, study)

    study._storage.remove_session()


def _run_trial(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    catch: tuple[type[Exception], ...],
) -> trial_module.FrozenTrial:
    if is_heartbeat_enabled(study._storage):
        optuna.storages.fail_stale_trials(study)

    trial = study.ask()

    state: TrialState | None = None
    value_or_values: float | Sequence[float] | None = None
    func_err: Exception | KeyboardInterrupt | None = None
    func_err_fail_exc_info: Any | None = None

    with get_heartbeat_thread(trial._trial_id, study._storage):
        try:
            value_or_values = func(trial)
        except exceptions.TrialPruned as e:
            # TODO(mamu): Handle multi-objective cases.
            state = TrialState.PRUNED
            func_err = e
        except (Exception, KeyboardInterrupt) as e:
            state = TrialState.FAIL
            func_err = e
            func_err_fail_exc_info = sys.exc_info()

    # `_tell_with_warning` may raise during trial post-processing.
    try:
        frozen_trial = _tell_with_warning(
            study=study,
            trial=trial,
            value_or_values=value_or_values,
            state=state,
            suppress_warning=True,
        )
    except Exception:
        frozen_trial = study._storage.get_trial(trial._trial_id)
        raise
    finally:
        if frozen_trial.state == TrialState.COMPLETE:
            study._log_completed_trial(frozen_trial)
        elif frozen_trial.state == TrialState.PRUNED:
            _logger.info("Trial {} pruned. {}".format(frozen_trial.number, str(func_err)))
        elif frozen_trial.state == TrialState.FAIL:
            if func_err is not None:
                _log_failed_trial(
                    frozen_trial,
                    repr(func_err),
                    exc_info=func_err_fail_exc_info,
                    value_or_values=value_or_values,
                )
            elif STUDY_TELL_WARNING_KEY in frozen_trial.system_attrs:
                _log_failed_trial(
                    frozen_trial,
                    frozen_trial.system_attrs[STUDY_TELL_WARNING_KEY],
                    value_or_values=value_or_values,
                )
            else:
                assert False, "Should not reach."
        else:
            assert False, "Should not reach."

    if (
        frozen_trial.state == TrialState.FAIL
        and func_err is not None
        and not isinstance(func_err, catch)
    ):
        raise func_err
    return frozen_trial


def _log_failed_trial(
    trial: FrozenTrial,
    message: str | Warning,
    exc_info: Any = None,
    value_or_values: Any = None,
) -> None:
    _logger.warning(
        "Trial {} failed with parameters: {} because of the following error: {}.".format(
            trial.number, trial.params, message
        ),
        exc_info=exc_info,
    )

    _logger.warning("Trial {} failed with value {}.".format(trial.number, repr(value_or_values)))
