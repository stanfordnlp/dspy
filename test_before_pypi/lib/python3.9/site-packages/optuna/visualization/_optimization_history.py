from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from enum import Enum
import math
from typing import cast
from typing import NamedTuple

import numpy as np

from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _ValueState(Enum):
    Feasible = 0
    Infeasible = 1
    Incomplete = 2


class _ValuesInfo(NamedTuple):
    values: list[float]
    stds: list[float] | None
    label_name: str
    states: list[_ValueState]


class _OptimizationHistoryInfo(NamedTuple):
    trial_numbers: list[int]
    values_info: _ValuesInfo
    best_values_info: _ValuesInfo | None


def _get_optimization_history_info_list(
    study: Study | Sequence[Study],
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
    error_bar: bool,
) -> list[_OptimizationHistoryInfo]:
    _check_plot_args(study, target, target_name)
    if isinstance(study, Study):
        studies = [study]
    else:
        studies = list(study)

    info_list: list[_OptimizationHistoryInfo] = []
    for study in studies:
        trials = study.get_trials()
        label_name = target_name if len(studies) == 1 else f"{target_name} of {study.study_name}"
        values = []
        value_states = []
        for trial in trials:
            if trial.state != TrialState.COMPLETE:
                values.append(float("nan"))
                value_states.append(_ValueState.Incomplete)
                continue
            constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
            if constraints is None or all([x <= 0.0 for x in constraints]):
                value_states.append(_ValueState.Feasible)
            else:
                value_states.append(_ValueState.Infeasible)
            if target is not None:
                values.append(target(trial))
            else:
                values.append(cast(float, trial.value))
        if target is not None:
            # We don't calculate best for user-defined target function since we cannot tell
            # which direction is better.
            best_values_info: _ValuesInfo | None = None
        else:
            feasible_best_values = []
            if study.direction == StudyDirection.MINIMIZE:
                feasible_best_values = [
                    v if s == _ValueState.Feasible else float("inf")
                    for v, s in zip(values, value_states)
                ]
                best_values = list(np.minimum.accumulate(feasible_best_values))
            else:
                feasible_best_values = [
                    v if s == _ValueState.Feasible else -float("inf")
                    for v, s in zip(values, value_states)
                ]
                best_values = list(np.maximum.accumulate(feasible_best_values))
            best_label_name = (
                "Best Value" if len(studies) == 1 else f"Best Value of {study.study_name}"
            )
            best_values_info = _ValuesInfo(best_values, None, best_label_name, value_states)
        info_list.append(
            _OptimizationHistoryInfo(
                trial_numbers=[t.number for t in trials],
                values_info=_ValuesInfo(values, None, label_name, value_states),
                best_values_info=best_values_info,
            )
        )

    if len(info_list) == 0:
        _logger.warning("There are no studies.")

    feasible_trial_count = sum(
        info.values_info.states.count(_ValueState.Feasible) for info in info_list
    )
    infeasible_trial_count = sum(
        info.values_info.states.count(_ValueState.Infeasible) for info in info_list
    )
    if feasible_trial_count + infeasible_trial_count == 0:
        _logger.warning("There are no complete trials.")
        info_list.clear()

    if not error_bar:
        return info_list

    # When error_bar=True, a list of 0 or 1 element is returned.
    if len(info_list) == 0:
        return []
    if feasible_trial_count == 0:
        _logger.warning("There are no feasible trials.")
        return []

    all_trial_numbers = [number for info in info_list for number in info.trial_numbers]
    max_num_trial = max(all_trial_numbers) + 1

    def _aggregate(label_name: str, use_best_value: bool) -> tuple[list[int], _ValuesInfo]:
        # Calculate mean and std of values for each trial number.
        values: list[list[float]] = [[] for _ in range(max_num_trial)]
        states: list[list[_ValueState]] = [[] for _ in range(max_num_trial)]
        assert info_list is not None
        for trial_numbers, values_info, best_values_info in info_list:
            if use_best_value:
                assert best_values_info is not None
                values_info = best_values_info
            for n, v, s in zip(trial_numbers, values_info.values, values_info.states):
                if not math.isinf(v):
                    if not use_best_value and s == _ValueState.Feasible:
                        values[n].append(v)
                    elif use_best_value:
                        values[n].append(v)
                states[n].append(s)
        trial_numbers_union: list[int] = []
        value_states: list[_ValueState] = []
        value_means: list[float] = []
        value_stds: list[float] = []
        for i in range(max_num_trial):
            if len(states[i]) > 0 and _ValueState.Feasible in states[i]:
                value_states.append(_ValueState.Feasible)
                trial_numbers_union.append(i)
                value_means.append(np.mean(values[i]).item())
                value_stds.append(np.std(values[i]).item())
            else:
                value_states.append(_ValueState.Infeasible)
        return trial_numbers_union, _ValuesInfo(value_means, value_stds, label_name, value_states)

    eb_trial_numbers, eb_values_info = _aggregate(target_name, False)
    eb_best_values_info: _ValuesInfo | None = None
    if target is None:
        _, eb_best_values_info = _aggregate("Best Value", True)
    return [_OptimizationHistoryInfo(eb_trial_numbers, eb_values_info, eb_best_values_info)]


def plot_optimization_history(
    study: Study | Sequence[Study],
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
    error_bar: bool = False,
) -> "go.Figure":
    """Plot optimization history of all trials in a study.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
            You can pass multiple studies if you want to compare those optimization histories.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label and the legend.
        error_bar:
            A flag to show the error bar.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()

    info_list = _get_optimization_history_info_list(study, target, target_name, error_bar)
    return _get_optimization_history_plot(info_list, target_name)


def _get_optimization_history_plot(
    info_list: list[_OptimizationHistoryInfo],
    target_name: str,
) -> "go.Figure":
    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "Trial"},
        yaxis={"title": target_name},
    )

    traces = []
    for trial_numbers, values_info, best_values_info in info_list:
        infeasible_trial_numbers = [
            n for n, s in zip(trial_numbers, values_info.states) if s == _ValueState.Infeasible
        ]
        if values_info.stds is None:
            error_y = None
            feasible_trial_numbers = [
                num
                for num, s in zip(trial_numbers, values_info.states)
                if s == _ValueState.Feasible
            ]
            feasible_trial_values = []
            for num in feasible_trial_numbers:
                feasible_trial_values.append(values_info.values[num])
            infeasible_trial_values = []
            for num in infeasible_trial_numbers:
                infeasible_trial_values.append(values_info.values[num])
        else:
            if (
                _ValueState.Infeasible in values_info.states
                or _ValueState.Incomplete in values_info.states
            ):
                _logger.warning(
                    "Your study contains infeasible trials. "
                    "In optimization history plot, "
                    "error bars are calculated for only feasible trial values."
                )
            error_y = {"type": "data", "array": values_info.stds, "visible": True}
            feasible_trial_numbers = trial_numbers
            feasible_trial_values = values_info.values
            infeasible_trial_values = []
        traces.append(
            go.Scatter(
                x=feasible_trial_numbers,
                y=feasible_trial_values,
                error_y=error_y,
                mode="markers",
                name=values_info.label_name,
            )
        )
        if best_values_info is not None:
            traces.append(
                go.Scatter(
                    x=trial_numbers,
                    y=best_values_info.values,
                    name=best_values_info.label_name,
                    mode="lines",
                )
            )
            if best_values_info.stds is not None:
                upper = np.array(best_values_info.values) + np.array(best_values_info.stds)
                traces.append(
                    go.Scatter(
                        x=trial_numbers,
                        y=upper,
                        mode="lines",
                        line=dict(width=0.01),
                        showlegend=False,
                    )
                )
                lower = np.array(best_values_info.values) - np.array(best_values_info.stds)
                traces.append(
                    go.Scatter(
                        x=trial_numbers,
                        y=lower,
                        mode="none",
                        showlegend=False,
                        fill="tonexty",
                        fillcolor="rgba(255,0,0,0.2)",
                    )
                )
        traces.append(
            go.Scatter(
                x=infeasible_trial_numbers,
                y=infeasible_trial_values,
                error_y=error_y,
                mode="markers",
                name="Infeasible Trial",
                marker={"color": "#cccccc"},
                showlegend=False,
            )
        )
    return go.Figure(data=traces, layout=layout)
