from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np

from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _constrained_dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    """Checks constrained-domination.

    A trial x is said to constrained-dominate a trial y, if any of the following conditions is
    true:
    1) Trial x is feasible and trial y is not.
    2) Trial x and y are both infeasible, but solution x has a smaller overall constraint
    violation.
    3) Trial x and y are feasible and trial x dominates trial y.
    """

    constraints0 = trial0.system_attrs.get(_CONSTRAINTS_KEY)
    constraints1 = trial1.system_attrs.get(_CONSTRAINTS_KEY)

    if constraints0 is None:
        warnings.warn(
            f"Trial {trial0.number} does not have constraint values."
            " It will be dominated by the other trials."
        )

    if constraints1 is None:
        warnings.warn(
            f"Trial {trial1.number} does not have constraint values."
            " It will be dominated by the other trials."
        )

    if constraints0 is None and constraints1 is None:
        # Neither Trial x nor y has constraints values
        return _dominates(trial0, trial1, directions)

    if constraints0 is not None and constraints1 is None:
        # Trial x has constraint values, but y doesn't.
        return True

    if constraints0 is None and constraints1 is not None:
        # If Trial y has constraint values, but x doesn't.
        return False

    assert isinstance(constraints0, (list, tuple))
    assert isinstance(constraints1, (list, tuple))

    if len(constraints0) != len(constraints1):
        raise ValueError("Trials with different numbers of constraints cannot be compared.")

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    satisfy_constraints0 = all(v <= 0 for v in constraints0)
    satisfy_constraints1 = all(v <= 0 for v in constraints1)

    if satisfy_constraints0 and satisfy_constraints1:
        # Both trials satisfy the constraints.
        return _dominates(trial0, trial1, directions)

    if satisfy_constraints0:
        # trial0 satisfies the constraints, but trial1 violates them.
        return True

    if satisfy_constraints1:
        # trial1 satisfies the constraints, but trial0 violates them.
        return False

    # Both trials violate the constraints.
    violation0 = sum(v for v in constraints0 if v > 0)
    violation1 = sum(v for v in constraints1 if v > 0)
    return violation0 < violation1


def _evaluate_penalty(population: Sequence[FrozenTrial]) -> np.ndarray:
    """Evaluate feasibility of trials in population.
    Returns:
        A list of feasibility status T/F/None of trials in population, where T/F means
        feasible/infeasible and None means that the trial does not have constraint values.
    """

    penalty: list[float] = []
    for trial in population:
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        if constraints is None:
            penalty.append(np.nan)
        else:
            penalty.append(sum(v for v in constraints if v > 0))
    return np.array(penalty)


def _validate_constraints(
    population: list[FrozenTrial],
    *,
    is_constrained: bool = False,
) -> None:
    if not is_constrained:
        return

    num_constraints = max(
        [len(t.system_attrs.get(_CONSTRAINTS_KEY, [])) for t in population], default=0
    )
    for _trial in population:
        _constraints = _trial.system_attrs.get(_CONSTRAINTS_KEY)
        if _constraints is None:
            warnings.warn(
                f"Trial {_trial.number} does not have constraint values."
                " It will be dominated by the other trials."
            )
            continue
        if np.any(np.isnan(np.array(_constraints))):
            raise ValueError("NaN is not acceptable as constraint value.")
        elif len(_constraints) != num_constraints:
            raise ValueError("Trials with different numbers of constraints cannot be compared.")
