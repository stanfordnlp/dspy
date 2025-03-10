from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any
from typing import NamedTuple
import warnings

import numpy as np

from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical
from optuna.visualization._utils import _is_reverse_scale


if _imports.is_successful():
    from optuna.visualization._plotly_imports import Contour
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import Scatter
    from optuna.visualization._utils import COLOR_SCALE

_logger = get_logger(__name__)


PADDING_RATIO = 0.05


class _AxisInfo(NamedTuple):
    name: str
    range: tuple[float, float]
    is_log: bool
    is_cat: bool
    indices: list[str | int | float]
    values: list[str | float | None]


class _SubContourInfo(NamedTuple):
    xaxis: _AxisInfo
    yaxis: _AxisInfo
    z_values: dict[tuple[int, int], float]
    constraints: list[bool] = []


class _ContourInfo(NamedTuple):
    sorted_params: list[str]
    sub_plot_infos: list[list[_SubContourInfo]]
    reverse_scale: bool
    target_name: str


class _PlotValues(NamedTuple):
    x: list[Any]
    y: list[Any]


def plot_contour(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the parameter relationship as contour plot in a study.

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
            Target's name to display on the color bar.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.

    .. note::
        The colormap is reversed when the ``target`` argument isn't :obj:`None` or ``direction``
        of :class:`~optuna.study.Study` is ``minimize``.
    """

    _imports.check()
    info = _get_contour_info(study, params, target, target_name)
    return _get_contour_plot(info)


def _get_contour_plot(info: _ContourInfo) -> "go.Figure":
    layout = go.Layout(title="Contour Plot")

    sorted_params = info.sorted_params
    sub_plot_infos = info.sub_plot_infos
    reverse_scale = info.reverse_scale
    target_name = info.target_name

    if len(sorted_params) <= 1:
        return go.Figure(data=[], layout=layout)

    if len(sorted_params) == 2:
        x_param = sorted_params[0]
        y_param = sorted_params[1]
        sub_plot_info = sub_plot_infos[0][0]
        sub_plots = _get_contour_subplot(sub_plot_info, reverse_scale, target_name)
        figure = go.Figure(data=sub_plots, layout=layout)
        figure.update_xaxes(title_text=x_param, range=sub_plot_info.xaxis.range)
        figure.update_yaxes(title_text=y_param, range=sub_plot_info.yaxis.range)

        if sub_plot_info.xaxis.is_cat:
            figure.update_xaxes(type="category")
        if sub_plot_info.yaxis.is_cat:
            figure.update_yaxes(type="category")

        if sub_plot_info.xaxis.is_log:
            log_range = [math.log10(p) for p in sub_plot_info.xaxis.range]
            figure.update_xaxes(range=log_range, type="log")
        if sub_plot_info.yaxis.is_log:
            log_range = [math.log10(p) for p in sub_plot_info.yaxis.range]
            figure.update_yaxes(range=log_range, type="log")
    else:
        figure = make_subplots(
            rows=len(sorted_params), cols=len(sorted_params), shared_xaxes=True, shared_yaxes=True
        )
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once.
        for x_i, x_param in enumerate(sorted_params):
            for y_i, y_param in enumerate(sorted_params):
                if x_param == y_param:
                    figure.add_trace(go.Scatter(), row=y_i + 1, col=x_i + 1)
                else:
                    sub_plots = _get_contour_subplot(
                        sub_plot_infos[y_i][x_i], reverse_scale, target_name
                    )
                    contour = sub_plots[0]
                    scatter = sub_plots[1]
                    contour.update(showscale=showscale)  # showscale's default is True.
                    if showscale:
                        showscale = False
                    figure.add_trace(contour, row=y_i + 1, col=x_i + 1)
                    figure.add_trace(scatter, row=y_i + 1, col=x_i + 1)

                xaxis = sub_plot_infos[y_i][x_i].xaxis
                yaxis = sub_plot_infos[y_i][x_i].yaxis
                figure.update_xaxes(range=xaxis.range, row=y_i + 1, col=x_i + 1)
                figure.update_yaxes(range=yaxis.range, row=y_i + 1, col=x_i + 1)

                if xaxis.is_cat:
                    figure.update_xaxes(type="category", row=y_i + 1, col=x_i + 1)
                if yaxis.is_cat:
                    figure.update_yaxes(type="category", row=y_i + 1, col=x_i + 1)

                if xaxis.is_log:
                    log_range = [math.log10(p) for p in xaxis.range]
                    figure.update_xaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)
                if yaxis.is_log:
                    log_range = [math.log10(p) for p in yaxis.range]
                    figure.update_yaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)

                if x_i == 0:
                    figure.update_yaxes(title_text=y_param, row=y_i + 1, col=x_i + 1)
                if y_i == len(sorted_params) - 1:
                    figure.update_xaxes(title_text=x_param, row=y_i + 1, col=x_i + 1)

    return figure


def _get_contour_subplot(
    info: _SubContourInfo,
    reverse_scale: bool,
    target_name: str = "Objective Value",
) -> tuple["Contour", "Scatter", "Scatter"]:
    x_indices = info.xaxis.indices
    y_indices = info.yaxis.indices

    if len(x_indices) < 2 or len(y_indices) < 2:
        return go.Contour(), go.Scatter(), go.Scatter()
    if len(info.z_values) == 0:
        warnings.warn(
            f"Contour plot will not be displayed because `{info.xaxis.name}` and "
            f"`{info.yaxis.name}` cannot co-exist in `trial.params`."
        )
        return go.Contour(), go.Scatter(), go.Scatter()

    feasible = _PlotValues([], [])
    infeasible = _PlotValues([], [])

    for x_value, y_value, c in zip(info.xaxis.values, info.yaxis.values, info.constraints):
        if x_value is not None and y_value is not None:
            if c:
                feasible.x.append(x_value)
                feasible.y.append(y_value)
            else:
                infeasible.x.append(x_value)
                infeasible.y.append(y_value)

    z_values = np.full((len(y_indices), len(x_indices)), np.nan)

    xys = np.array(list(info.z_values.keys()))
    zs = np.array(list(info.z_values.values()))

    z_values[xys[:, 1], xys[:, 0]] = zs

    contour = go.Contour(
        x=x_indices,
        y=y_indices,
        z=z_values,
        colorbar={"title": target_name},
        colorscale=COLOR_SCALE,
        connectgaps=True,
        contours_coloring="heatmap",
        hoverinfo="none",
        line_smoothing=1.3,
        reversescale=reverse_scale,
    )

    return (
        contour,
        _create_scatter(feasible.x, feasible.y, is_feasible=True),
        _create_scatter(infeasible.x, infeasible.y, is_feasible=False),
    )


def _create_scatter(x: list[Any], y: list[Any], is_feasible: bool) -> Scatter:
    edge_color = "Gray"
    marker_color = "black" if is_feasible else "#cccccc"
    name = "Feasible Trial" if is_feasible else "Infeasible Trial"
    return go.Scatter(
        x=x,
        y=y,
        marker={
            "line": {"width": 2.0, "color": edge_color},
            "color": marker_color,
        },
        mode="markers",
        name=name,
        showlegend=False,
    )


def _get_contour_info(
    study: Study,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> _ContourInfo:
    _check_plot_args(study, target, target_name)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        sorted_params = []
    elif params is None:
        sorted_params = sorted(all_params)
    else:
        if len(params) <= 1:
            _logger.warning("The length of params must be greater than 1.")

        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(set(params))

    sub_plot_infos: list[list[_SubContourInfo]]
    if len(sorted_params) == 2:
        x_param = sorted_params[0]
        y_param = sorted_params[1]
        sub_plot_info = _get_contour_subplot_info(study, trials, x_param, y_param, target)
        sub_plot_infos = [[sub_plot_info]]
    else:
        sub_plot_infos = []
        for i, y_param in enumerate(sorted_params):
            sub_plot_infos.append([])
            for x_param in sorted_params:
                sub_plot_info = _get_contour_subplot_info(study, trials, x_param, y_param, target)
                sub_plot_infos[i].append(sub_plot_info)

    reverse_scale = _is_reverse_scale(study, target)

    return _ContourInfo(
        sorted_params=sorted_params,
        sub_plot_infos=sub_plot_infos,
        reverse_scale=reverse_scale,
        target_name=target_name,
    )


def _get_contour_subplot_info(
    study: Study,
    trials: list[FrozenTrial],
    x_param: str,
    y_param: str,
    target: Callable[[FrozenTrial], float] | None,
) -> _SubContourInfo:
    xaxis = _get_axis_info(trials, x_param)
    yaxis = _get_axis_info(trials, y_param)

    if x_param == y_param:
        return _SubContourInfo(xaxis=xaxis, yaxis=yaxis, z_values={})

    if len(xaxis.indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(x_param))
        return _SubContourInfo(xaxis=xaxis, yaxis=yaxis, z_values={})
    if len(yaxis.indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(y_param))
        return _SubContourInfo(xaxis=xaxis, yaxis=yaxis, z_values={})

    z_values: dict[tuple[int, int], float] = {}
    for i, trial in enumerate(trials):
        if x_param not in trial.params or y_param not in trial.params:
            continue
        x_value = xaxis.values[i]
        y_value = yaxis.values[i]
        assert x_value is not None
        assert y_value is not None
        x_i = xaxis.indices.index(x_value)
        y_i = yaxis.indices.index(y_value)

        if target is None:
            value = trial.value
        else:
            value = target(trial)
        assert value is not None

        existing = z_values.get((x_i, y_i))
        if existing is None or target is not None:
            # When target function is present, we can't be sure what the z-value
            # represents and therefore we don't know how to select the best one.
            z_values[(x_i, y_i)] = value
        else:
            z_values[(x_i, y_i)] = (
                min(existing, value)
                if study.direction is StudyDirection.MINIMIZE
                else max(existing, value)
            )

    return _SubContourInfo(
        xaxis=xaxis,
        yaxis=yaxis,
        z_values=z_values,
        constraints=[_satisfy_constraints(t) for t in trials],
    )


def _satisfy_constraints(trial: FrozenTrial) -> bool:
    constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
    return constraints is None or all([x <= 0.0 for x in constraints])


def _get_axis_info(trials: list[FrozenTrial], param_name: str) -> _AxisInfo:
    values: list[str | float | None]
    if _is_numerical(trials, param_name):
        values = [t.params.get(param_name) for t in trials]
    else:
        values = [
            str(t.params.get(param_name)) if param_name in t.params else None for t in trials
        ]

    min_value = min([v for v in values if v is not None])
    max_value = max([v for v in values if v is not None])

    if _is_log_scale(trials, param_name):
        min_value = float(min_value)
        max_value = float(max_value)
        padding = (math.log10(max_value) - math.log10(min_value)) * PADDING_RATIO
        min_value = math.pow(10, math.log10(min_value) - padding)
        max_value = math.pow(10, math.log10(max_value) + padding)
        is_log = True
        is_cat = False

    elif _is_numerical(trials, param_name):
        min_value = float(min_value)
        max_value = float(max_value)
        padding = (max_value - min_value) * PADDING_RATIO
        min_value = min_value - padding
        max_value = max_value + padding
        is_log = False
        is_cat = False

    else:
        unique_values = set(values)
        span = len(unique_values) - 1
        if None in unique_values:
            span -= 1
        padding = span * PADDING_RATIO
        min_value = -padding
        max_value = span + padding
        is_log = False
        is_cat = True

    indices = sorted(set([v for v in values if v is not None]))

    if len(indices) < 2:
        return _AxisInfo(
            name=param_name,
            range=(min_value, max_value),
            is_log=is_log,
            is_cat=is_cat,
            indices=indices,
            values=values,
        )

    if _is_numerical(trials, param_name):
        indices.insert(0, min_value)
        indices.append(max_value)

    return _AxisInfo(
        name=param_name,
        range=(min_value, max_value),
        is_log=is_log,
        is_cat=is_cat,
        indices=indices,
        values=values,
    )
