from __future__ import annotations

import logging
from typing import Any
from typing import TYPE_CHECKING
import warnings

from tqdm.auto import tqdm

from optuna import logging as optuna_logging


if TYPE_CHECKING:
    from optuna.study import Study

_tqdm_handler: _TqdmLoggingHandler | None = None


# Reference: https://gist.github.com/hvy/8b80c2cedf02b15c24f85d1fa17ebe02
class _TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: Any) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class _ProgressBar:
    """Progress Bar implementation for :func:`~optuna.study.Study.optimize` on the top of `tqdm`.

    Args:
        is_valid:
            Whether to show progress bars in :func:`~optuna.study.Study.optimize`.
        n_trials:
            The number of trials.
        timeout:
            Stop study after the given number of second(s).
    """

    def __init__(
        self,
        is_valid: bool,
        n_trials: int | None = None,
        timeout: float | None = None,
    ) -> None:
        if is_valid and n_trials is None and timeout is None:
            warnings.warn("Progress bar won't be displayed because n_trials and timeout are None.")

        self._is_valid = is_valid and (n_trials or timeout) is not None
        self._n_trials = n_trials
        self._timeout = timeout
        self._last_elapsed_seconds = 0.0

        if self._is_valid:
            if self._n_trials is not None:
                self._progress_bar = tqdm(total=self._n_trials)
            elif self._timeout is not None:
                total = tqdm.format_interval(self._timeout)
                fmt = "{desc} {percentage:3.0f}%|{bar}| {elapsed}/" + total
                self._progress_bar = tqdm(total=self._timeout, bar_format=fmt)
            else:
                assert False

            global _tqdm_handler

            _tqdm_handler = _TqdmLoggingHandler()
            _tqdm_handler.setLevel(logging.INFO)
            _tqdm_handler.setFormatter(optuna_logging.create_default_formatter())
            optuna_logging.disable_default_handler()
            optuna_logging._get_library_root_logger().addHandler(_tqdm_handler)

    def update(self, elapsed_seconds: float, study: Study) -> None:
        """Update the progress bars if ``is_valid`` is :obj:`True`.

        Args:
            elapsed_seconds:
                The time past since :func:`~optuna.study.Study.optimize` started.
            study:
                The current study object.
        """

        if self._is_valid:
            if not study._is_multi_objective():
                # Not updating the progress bar when there are no complete trial.
                try:
                    msg = (
                        f"Best trial: {study.best_trial.number}. "
                        f"Best value: {study.best_value:.6g}"
                    )

                    self._progress_bar.set_description(msg)
                except ValueError:
                    pass

            if self._n_trials is not None:
                self._progress_bar.update(1)
                if self._timeout is not None:
                    self._progress_bar.set_postfix_str(
                        "{:.02f}/{} seconds".format(elapsed_seconds, self._timeout)
                    )

            elif self._timeout is not None:
                time_diff = elapsed_seconds - self._last_elapsed_seconds
                if elapsed_seconds > self._timeout:
                    # Clip elapsed time to avoid tqdm warnings.
                    time_diff -= elapsed_seconds - self._timeout

                self._progress_bar.update(time_diff)
                self._last_elapsed_seconds = elapsed_seconds

            else:
                assert False

    def close(self) -> None:
        """Close progress bars."""

        if self._is_valid:
            self._progress_bar.close()
            assert _tqdm_handler is not None
            optuna_logging._get_library_root_logger().removeHandler(_tqdm_handler)
            optuna_logging.enable_default_handler()
