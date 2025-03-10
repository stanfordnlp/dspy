from __future__ import annotations

from collections.abc import Container

import optuna
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class MaxTrialsCallback:
    """Set a maximum number of trials before ending the study.

    While the ``n_trials`` argument of :meth:`optuna.study.Study.optimize` sets the number of
    trials that will be run, you may want to continue running until you have a certain number of
    successfully completed trials or stop the study when you have a certain number of trials that
    fail. This ``MaxTrialsCallback`` class allows you to set a maximum number of trials for a
    particular :class:`~optuna.trial.TrialState` before stopping the study.

    Example:

        .. testcode::

            import optuna
            from optuna.study import MaxTrialsCallback
            from optuna.trial import TrialState


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                return x**2


            study = optuna.create_study()
            study.optimize(
                objective,
                callbacks=[MaxTrialsCallback(10, states=(TrialState.COMPLETE,))],
            )

    Args:
        n_trials:
            The max number of trials. Must be set to an integer.
        states:
            Tuple of the :class:`~optuna.trial.TrialState` to be counted
            towards the max trials limit. Default value is ``(TrialState.COMPLETE,)``.
            If :obj:`None`, count all states.
    """

    def __init__(
        self, n_trials: int, states: Container[TrialState] | None = (TrialState.COMPLETE,)
    ) -> None:
        self._n_trials = n_trials
        self._states = states

    def __call__(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        trials = study.get_trials(deepcopy=False, states=self._states)
        n_complete = len(trials)
        if n_complete >= self._n_trials:
            study.stop()
