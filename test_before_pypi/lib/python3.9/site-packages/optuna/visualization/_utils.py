from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import json
from typing import Any
from typing import cast
import warnings

import numpy as np

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.visualization import _plotly_imports


__all__ = ["is_available"]
_logger = optuna.logging.get_logger(__name__)


def is_available() -> bool:
    """Returns whether visualization with plotly is available or not.

    .. note::

        :mod:`~optuna.visualization` module depends on plotly version 4.0.0 or higher. If a
        supported version of plotly isn't installed in your environment, this function will return
        :obj:`False`. In such case, please execute ``$ pip install -U plotly>=4.0.0`` to install
        plotly.

    Returns:
        :obj:`True` if visualization with plotly is available, :obj:`False` otherwise.
    """

    return _plotly_imports._imports.is_successful()


if is_available():
    import plotly.colors

    COLOR_SCALE = plotly.colors.sequential.Blues


def _check_plot_args(
    study: Study | Sequence[Study],
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
) -> None:
    studies: Sequence[Study]
    if isinstance(study, Study):
        studies = [study]
    else:
        studies = study

    if target is None and any(study._is_multi_objective() for study in studies):
        raise ValueError(
            "If the `study` is being used for multi-objective optimization, "
            "please specify the `target`."
        )

    if target is not None and target_name == "Objective Value":
        warnings.warn(
            "`target` is specified, but `target_name` is the default value, 'Objective Value'."
        )


def _is_log_scale(trials: list[FrozenTrial], param: str) -> bool:
    for trial in trials:
        if param not in trial.params:
            continue
        dist = trial.distributions[param]
        return isinstance(dist, (FloatDistribution, IntDistribution)) and dist.log
    return False


def _is_numerical(trials: list[FrozenTrial], param: str) -> bool:
    for trial in trials:
        if param not in trial.params:
            continue
        dist = trial.distributions[param]
        if isinstance(dist, (IntDistribution, FloatDistribution)):
            return True
        elif isinstance(dist, CategoricalDistribution):
            # NOTE: Although it is a bit odd to do so, we keep it as is only for visualization.
            return all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in dist.choices
            )
        else:
            assert False, "Should not reach."
    return True


def _get_param_values(trials: list[FrozenTrial], p_name: str) -> list[Any]:
    values = [t.params[p_name] for t in trials if p_name in t.params]
    if _is_numerical(trials, p_name):
        return values
    return list(map(str, values))


def _get_skipped_trial_numbers(
    trials: list[FrozenTrial], used_param_names: Sequence[str]
) -> set[int]:
    """Utility function for ``plot_parallel_coordinate``.

    If trial's parameters do not contain a parameter in ``used_param_names``,
    ``plot_parallel_coordinate`` methods do not use such trials.

    Args:
        trials:
            List of ``FrozenTrial``s.
        used_param_names:
            The parameter names used in ``plot_parallel_coordinate``.

    Returns:
        A set of invalid trial numbers.
    """

    skipped_trial_numbers = set()
    for trial in trials:
        for used_param in used_param_names:
            if used_param not in trial.params.keys():
                skipped_trial_numbers.add(trial.number)
                break
    return skipped_trial_numbers


def _filter_nonfinite(
    trials: list[FrozenTrial],
    target: Callable[[FrozenTrial], float] | None = None,
    with_message: bool = True,
) -> list[FrozenTrial]:
    # For multi-objective optimization target must be specified to select
    # one of objective values to filter trials by (and plot by later on).
    # This function is not raising when target is missing, since we're
    # assuming plot args have been sanitized before.
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    filtered_trials: list[FrozenTrial] = []
    for trial in trials:
        value = target(trial)

        try:
            value = float(value)
        except (
            ValueError,
            TypeError,
        ):
            warnings.warn(
                f"Trial{trial.number}'s target value {repr(value)} could not be cast to float."
            )
            raise

        # Not a Number, positive infinity and negative infinity are considered to be non-finite.
        if not np.isfinite(value):
            if with_message:
                _logger.warning(
                    f"Trial {trial.number} is omitted in visualization "
                    "because its objective value is inf or nan."
                )
        else:
            filtered_trials.append(trial)

    return filtered_trials


def _is_reverse_scale(study: Study, target: Callable[[FrozenTrial], float] | None) -> bool:
    return target is not None or study.direction == StudyDirection.MINIMIZE


def _make_json_compatible(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        # The value can't be converted to JSON directly, so return a string representation.
        return str(value)


def _make_hovertext(trial: FrozenTrial) -> str:
    user_attrs = {key: _make_json_compatible(value) for key, value in trial.user_attrs.items()}
    user_attrs_dict = {"user_attrs": user_attrs} if user_attrs else {}
    text = json.dumps(
        {
            "number": trial.number,
            "values": trial.values,
            "params": trial.params,
            **user_attrs_dict,
        },
        indent=2,
    )
    return text.replace("\n", "<br>")
