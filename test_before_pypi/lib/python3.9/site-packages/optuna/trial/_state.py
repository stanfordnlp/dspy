import enum


class TrialState(enum.IntEnum):
    """State of a :class:`~optuna.trial.Trial`.

    Attributes:
        RUNNING:
            The :class:`~optuna.trial.Trial` is running.
        WAITING:
            The :class:`~optuna.trial.Trial` is waiting and unfinished.
        COMPLETE:
            The :class:`~optuna.trial.Trial` has been finished without any error.
        PRUNED:
            The :class:`~optuna.trial.Trial` has been pruned with
            :class:`~optuna.exceptions.TrialPruned`.
        FAIL:
            The :class:`~optuna.trial.Trial` has failed due to an uncaught error.
    """

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3
    WAITING = 4

    def __repr__(self) -> str:
        return str(self)

    def is_finished(self) -> bool:
        """Return a bool value to represent whether the trial state is unfinished or not.

        The unfinished state is either ``RUNNING`` or ``WAITING``.
        """

        return self != TrialState.RUNNING and self != TrialState.WAITING
