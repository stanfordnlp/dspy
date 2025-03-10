from __future__ import annotations

from collections.abc import Sequence

from optuna.trial import FrozenTrial


_CONSTRAINTS_KEY = "constraints"


def _get_feasible_trials(trials: Sequence[FrozenTrial]) -> list[FrozenTrial]:
    """Return feasible trials from given trials.

    This function assumes that the trials were created in constrained optimization.
    Therefore, if there is no violation value in the trial, it is considered infeasible.


    Returns:
        A list of feasible trials.
    """

    feasible_trials = []
    for trial in trials:
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        if constraints is not None and all(x <= 0.0 for x in constraints):
            feasible_trials.append(trial)
    return feasible_trials
