from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _create_frozen_trial(
    number: int = 0,
    values: Sequence[float] | None = None,
    constraints: Sequence[float] | None = None,
    params: dict[str, Any] | None = None,
    param_distributions: dict[str, BaseDistribution] | None = None,
    state: TrialState = TrialState.COMPLETE,
) -> optuna.trial.FrozenTrial:
    return FrozenTrial(
        number=number,
        value=1.0 if values is None else None,
        values=values,
        state=state,
        user_attrs={},
        system_attrs={} if constraints is None else {_CONSTRAINTS_KEY: list(constraints)},
        params=params or {},
        distributions=param_distributions or {},
        intermediate_values={},
        datetime_start=None,
        datetime_complete=None,
        trial_id=number,
    )
