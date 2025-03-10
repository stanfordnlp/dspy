from __future__ import annotations

from typing import NamedTuple

from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _TrialInfo(NamedTuple):
    trial_number: int
    sorted_intermediate_values: list[tuple[int, float]]
    feasible: bool


class _IntermediatePlotInfo(NamedTuple):
    trial_infos: list[_TrialInfo]


def _get_intermediate_plot_info(study: Study) -> _IntermediatePlotInfo:
    trials = study.get_trials(
        deepcopy=False, states=(TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING)
    )

    def _satisfies_constraints(trial: FrozenTrial) -> bool:
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        return constraints is None or all([x <= 0.0 for x in constraints])

    trial_infos = [
        _TrialInfo(
            trial.number, sorted(trial.intermediate_values.items()), _satisfies_constraints(trial)
        )
        for trial in trials
        if len(trial.intermediate_values) > 0
    ]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
    elif len(trial_infos) == 0:
        _logger.warning(
            "You need to set up the pruning feature to utilize `plot_intermediate_values()`"
        )

    return _IntermediatePlotInfo(trial_infos)


def plot_intermediate_values(study: Study) -> "go.Figure":
    """Plot intermediate values of all trials in a study.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()
    return _get_intermediate_plot(_get_intermediate_plot_info(study))


def _get_intermediate_plot(info: _IntermediatePlotInfo) -> "go.Figure":
    layout = go.Layout(
        title="Intermediate Values Plot",
        xaxis={"title": "Step"},
        yaxis={"title": "Intermediate Value"},
        showlegend=False,
    )

    trial_infos = info.trial_infos

    if len(trial_infos) == 0:
        return go.Figure(data=[], layout=layout)

    default_marker = {"maxdisplayed": 10}

    traces = [
        go.Scatter(
            x=tuple((x for x, _ in tinfo.sorted_intermediate_values)),
            y=tuple((y for _, y in tinfo.sorted_intermediate_values)),
            mode="lines+markers",
            marker=(
                default_marker
                if tinfo.feasible
                else {**default_marker, "color": "#CCCCCC"}  # type: ignore[dict-item]
            ),
            name="Trial{}".format(tinfo.trial_number),
        )
        for tinfo in trial_infos
    ]

    return go.Figure(data=traces, layout=layout)
