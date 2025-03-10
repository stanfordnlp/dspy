from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import cast
from typing import NamedTuple

from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._utils import _is_log_scale


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import Scatter
    from optuna.visualization._utils import COLOR_SCALE

_logger = get_logger(__name__)


class _SliceSubplotInfo(NamedTuple):
    param_name: str
    x: list[Any]
    y: list[float]
    trial_numbers: list[int]
    is_log: bool
    is_numerical: bool
    constraints: list[bool]
    x_labels: tuple[CategoricalChoiceType, ...] | None


class _SlicePlotInfo(NamedTuple):
    target_name: str
    subplots: list[_SliceSubplotInfo]


class _PlotValues(NamedTuple):
    x: list[Any]
    y: list[float]
    trial_numbers: list[int]


def _get_slice_subplot_info(
    trials: list[FrozenTrial],
    param: str,
    target: Callable[[FrozenTrial], float] | None,
    log_scale: bool,
    numerical: bool,
    x_labels: tuple[CategoricalChoiceType, ...] | None,
) -> _SliceSubplotInfo:
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    plot_info = _SliceSubplotInfo(
        param_name=param,
        x=[],
        y=[],
        trial_numbers=[],
        is_log=log_scale,
        is_numerical=numerical,
        x_labels=x_labels,
        constraints=[],
    )

    for t in trials:
        if param not in t.params:
            continue
        plot_info.x.append(t.params[param])
        plot_info.y.append(target(t))
        plot_info.trial_numbers.append(t.number)
        constraints = t.system_attrs.get(_CONSTRAINTS_KEY)
        plot_info.constraints.append(constraints is None or all([x <= 0.0 for x in constraints]))

    return plot_info


def _get_slice_plot_info(
    study: Study,
    params: list[str] | None,
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
) -> _SlicePlotInfo:
    _check_plot_args(study, target, target_name)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return _SlicePlotInfo(target_name, [])

    all_params = {p_name for t in trials for p_name in t.params.keys()}

    distributions = {}
    for trial in trials:
        for param_name, distribution in trial.distributions.items():
            if param_name not in distributions:
                distributions[param_name] = distribution

    x_labels = {}
    for param_name, distribution in distributions.items():
        if isinstance(distribution, CategoricalDistribution):
            x_labels[param_name] = distribution.choices

    if params is None:
        sorted_params = sorted(all_params)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError(f"Parameter {input_p_name} does not exist in your study.")
        sorted_params = sorted(set(params))

    return _SlicePlotInfo(
        target_name=target_name,
        subplots=[
            _get_slice_subplot_info(
                trials=trials,
                param=param,
                target=target,
                log_scale=_is_log_scale(trials, param),
                numerical=not isinstance(distributions[param], CategoricalDistribution),
                x_labels=x_labels.get(param),
            )
            for param in sorted_params
        ],
    )


def plot_slice(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the parameter relationship as slice plot in a study.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        params:
            Parameter list to visualize. The default is all parameters.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()
    return _get_slice_plot(_get_slice_plot_info(study, params, target, target_name))


def _get_slice_plot(info: _SlicePlotInfo) -> "go.Figure":
    layout = go.Layout(title="Slice Plot")

    if len(info.subplots) == 0:
        return go.Figure(data=[], layout=layout)
    elif len(info.subplots) == 1:
        figure = go.Figure(data=_generate_slice_subplot(info.subplots[0]), layout=layout)
        figure.update_xaxes(title_text=info.subplots[0].param_name)
        figure.update_yaxes(title_text=info.target_name)
        if not info.subplots[0].is_numerical:
            figure.update_xaxes(
                type="category", categoryorder="array", categoryarray=info.subplots[0].x_labels
            )
        elif info.subplots[0].is_log:
            figure.update_xaxes(type="log")
    else:
        figure = make_subplots(rows=1, cols=len(info.subplots), shared_yaxes=True)
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once.
        for column_index, subplot_info in enumerate(info.subplots, start=1):
            trace = _generate_slice_subplot(subplot_info)
            trace[0].update(marker={"showscale": showscale})  # showscale's default is True.
            if showscale:
                showscale = False
            for t in trace:
                figure.add_trace(t, row=1, col=column_index)
            figure.update_xaxes(title_text=subplot_info.param_name, row=1, col=column_index)
            if column_index == 1:
                figure.update_yaxes(title_text=info.target_name, row=1, col=column_index)
            if not subplot_info.is_numerical:
                figure.update_xaxes(
                    type="category",
                    categoryorder="array",
                    categoryarray=subplot_info.x_labels,
                    row=1,
                    col=column_index,
                )
            elif subplot_info.is_log:
                figure.update_xaxes(type="log", row=1, col=column_index)
        if len(info.subplots) > 3:
            # Ensure that each subplot has a minimum width without relying on autusizing.
            figure.update_layout(width=300 * len(info.subplots))

    return figure


def _generate_slice_subplot(subplot_info: _SliceSubplotInfo) -> list[Scatter]:
    trace = []

    feasible = _PlotValues([], [], [])
    infeasible = _PlotValues([], [], [])

    for x, y, num, c in zip(
        subplot_info.x, subplot_info.y, subplot_info.trial_numbers, subplot_info.constraints
    ):
        if x is not None or x != "None" or y is not None or y != "None":
            if c:
                feasible.x.append(x)
                feasible.y.append(y)
                feasible.trial_numbers.append(num)
            else:
                infeasible.x.append(x)
                infeasible.y.append(y)
    trace.append(
        go.Scatter(
            x=feasible.x,
            y=feasible.y,
            mode="markers",
            name="Feasible Trial",
            marker={
                "line": {"width": 0.5, "color": "Grey"},
                "color": feasible.trial_numbers,
                "colorscale": COLOR_SCALE,
                "colorbar": {
                    "title": "Trial",
                    "x": 1.0,  # Offset the colorbar position with a fixed width `xpad`.
                    "xpad": 40,
                },
            },
            showlegend=False,
        )
    )
    if len(infeasible.x) > 0:
        trace.append(
            go.Scatter(
                x=infeasible.x,
                y=infeasible.y,
                mode="markers",
                name="Infeasible Trial",
                marker={
                    "color": "#cccccc",
                },
                showlegend=False,
            )
        )

    return trace
