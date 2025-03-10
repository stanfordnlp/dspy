from __future__ import annotations

from collections.abc import Callable
import math
import typing
from typing import Any
from typing import NamedTuple

import numpy as np

from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical
from optuna.visualization.matplotlib._matplotlib_imports import _imports as matplotlib_imports


plotly_is_available = _imports.is_successful()
if plotly_is_available:
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import plotly
    from optuna.visualization._plotly_imports import Scatter
if matplotlib_imports.is_successful():
    # TODO(c-bata): Refactor to remove matplotlib and plotly dependencies in `_get_rank_info()`.
    # See https://github.com/optuna/optuna/pull/5133#discussion_r1414761672 for the discussion.
    from optuna.visualization.matplotlib._matplotlib_imports import plt as matplotlib_plt

_logger = get_logger(__name__)


PADDING_RATIO = 0.05


class _AxisInfo(NamedTuple):
    name: str
    range: tuple[float, float]
    is_log: bool
    is_cat: bool


class _RankSubplotInfo(NamedTuple):
    xaxis: _AxisInfo
    yaxis: _AxisInfo
    xs: list[Any]
    ys: list[Any]
    trials: list[FrozenTrial]
    zs: np.ndarray
    colors: np.ndarray


class _RankPlotInfo(NamedTuple):
    params: list[str]
    sub_plot_infos: list[list[_RankSubplotInfo]]
    target_name: str
    zs: np.ndarray
    colors: np.ndarray
    has_custom_target: bool


def plot_rank(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot parameter relations as scatter plots with colors indicating ranks of target value.

    Note that trials missing the specified parameters will not be plotted.

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
        This function requires plotly >= 5.0.0.
    """

    _imports.check()
    info = _get_rank_info(study, params, target, target_name)
    return _get_rank_plot(info)


def _get_order_with_same_order_averaging(data: np.ndarray) -> np.ndarray:
    order = np.zeros_like(data, dtype=float)
    data_sorted = np.sort(data)
    for i, d in enumerate(data):
        indices = np.where(data_sorted == d)[0]
        order[i] = sum(indices) / len(indices)
    return order


def _get_rank_info(
    study: Study,
    params: list[str] | None,
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
) -> _RankPlotInfo:
    _check_plot_args(study, target, target_name)

    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        params = []
    elif params is None:
        params = sorted(all_params)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))

    if len(params) == 0:
        _logger.warning("params is an empty list.")

    has_custom_target = True
    if target is None:

        def target(trial: FrozenTrial) -> float:
            return typing.cast(float, trial.value)

        has_custom_target = False
    target_values = np.array([target(trial) for trial in trials])
    raw_ranks = _get_order_with_same_order_averaging(target_values)
    color_idxs = raw_ranks / (len(trials) - 1) if len(trials) >= 2 else np.array([0.5])
    colors = _convert_color_idxs_to_scaled_rgb_colors(color_idxs)

    sub_plot_infos: list[list[_RankSubplotInfo]]
    if len(params) == 2:
        x_param = params[0]
        y_param = params[1]
        sub_plot_info = _get_rank_subplot_info(trials, target_values, colors, x_param, y_param)
        sub_plot_infos = [[sub_plot_info]]
    else:
        sub_plot_infos = [
            [
                _get_rank_subplot_info(trials, target_values, colors, x_param, y_param)
                for x_param in params
            ]
            for y_param in params
        ]

    return _RankPlotInfo(
        params=params,
        sub_plot_infos=sub_plot_infos,
        target_name=target_name,
        zs=target_values,
        colors=colors,
        has_custom_target=has_custom_target,
    )


def _get_rank_subplot_info(
    trials: list[FrozenTrial],
    target_values: np.ndarray,
    colors: np.ndarray,
    x_param: str,
    y_param: str,
) -> _RankSubplotInfo:
    xaxis = _get_axis_info(trials, x_param)
    yaxis = _get_axis_info(trials, y_param)

    infeasible_trial_ids = []
    filtered_ids = []
    for idx, trial in enumerate(trials):
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        if constraints is not None and any([x > 0.0 for x in constraints]):
            infeasible_trial_ids.append(idx)
        if x_param in trial.params and y_param in trial.params:
            filtered_ids.append(idx)

    filtered_trials = [trials[i] for i in filtered_ids]
    xs = [trial.params[x_param] for trial in filtered_trials]
    ys = [trial.params[y_param] for trial in filtered_trials]
    zs = target_values[filtered_ids]

    colors[infeasible_trial_ids] = (204, 204, 204)
    colors = colors[filtered_ids]
    return _RankSubplotInfo(
        xaxis=xaxis,
        yaxis=yaxis,
        xs=xs,
        ys=ys,
        trials=filtered_trials,
        zs=np.array(zs),
        colors=colors,
    )


def _get_axis_info(trials: list[FrozenTrial], param_name: str) -> _AxisInfo:
    values: list[str | float | None]
    is_numerical = _is_numerical(trials, param_name)
    if is_numerical:
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

    elif is_numerical:
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

    return _AxisInfo(
        name=param_name,
        range=(min_value, max_value),
        is_log=is_log,
        is_cat=is_cat,
    )


def _get_rank_subplot(
    info: _RankSubplotInfo, target_name: str, print_raw_objectives: bool
) -> "Scatter":
    def get_hover_text(trial: FrozenTrial, target_value: float) -> str:
        lines = [f"Trial #{trial.number}"]
        lines += [f"{k}: {v}" for k, v in trial.params.items()]
        lines += [f"<b>{target_name}: {target_value}</b>"]
        if print_raw_objectives:
            lines += [f"Objective #{i}: {v}" for i, v in enumerate(trial.values)]
        return "<br>".join(lines)

    scatter = go.Scatter(
        x=[str(x) for x in info.xs] if info.xaxis.is_cat else info.xs,
        y=[str(y) for y in info.ys] if info.yaxis.is_cat else info.ys,
        marker={
            "color": list(map(plotly.colors.label_rgb, info.colors)),
            "line": {"width": 0.5, "color": "Grey"},
        },
        mode="markers",
        showlegend=False,
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=[
            get_hover_text(trial, target_value)
            for trial, target_value in zip(info.trials, info.zs)
        ],
    )
    return scatter


class _TickInfo(NamedTuple):
    coloridxs: list[float]
    text: list[str]


def _get_tick_info(target_values: np.ndarray) -> _TickInfo:
    sorted_target_values = np.sort(target_values)
    coloridxs = [0, 0.25, 0.5, 0.75, 1]
    values = np.quantile(sorted_target_values, coloridxs)
    rank_text = ["min.", "25%", "50%", "75%", "max."]
    text = [f"{rank_text[i]} ({values[i]:3g})" for i in range(len(values))]
    return _TickInfo(coloridxs=coloridxs, text=text)


def _get_rank_plot(
    info: _RankPlotInfo,
) -> "go.Figure":
    params = info.params
    sub_plot_infos = info.sub_plot_infos

    layout = go.Layout(title=f"Rank ({info.target_name})")

    if len(params) == 0:
        return go.Figure(data=[], layout=layout)
    if len(params) == 2:
        x_param = params[0]
        y_param = params[1]
        sub_plot_info = sub_plot_infos[0][0]
        sub_plots = _get_rank_subplot(sub_plot_info, info.target_name, info.has_custom_target)

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
            rows=len(params),
            cols=len(params),
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.08 / len(params),
            vertical_spacing=0.08 / len(params),
        )

        figure.update_layout(layout)
        for x_i, x_param in enumerate(params):
            for y_i, y_param in enumerate(params):
                scatter = _get_rank_subplot(
                    sub_plot_infos[y_i][x_i], info.target_name, info.has_custom_target
                )
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
                if y_i == len(params) - 1:
                    figure.update_xaxes(title_text=x_param, row=y_i + 1, col=x_i + 1)

    tick_info = _get_tick_info(info.zs)

    colormap = "RdYlBu_r"
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=colormap,
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(thickness=10, tickvals=tick_info.coloridxs, ticktext=tick_info.text),
        ),
        hoverinfo="none",
        showlegend=False,
    )
    figure.add_trace(colorbar_trace)
    return figure


def _convert_color_idxs_to_scaled_rgb_colors(color_idxs: np.ndarray) -> np.ndarray:
    colormap = "RdYlBu_r"
    if plotly_is_available:
        # sample_colorscale requires plotly >= 5.0.0.
        labeled_colors = plotly.colors.sample_colorscale(colormap, color_idxs)
        scaled_rgb_colors = np.array([plotly.colors.unlabel_rgb(cl) for cl in labeled_colors])
        return scaled_rgb_colors
    else:
        cmap = matplotlib_plt.get_cmap(colormap)
        colors = cmap(color_idxs)[:, :3]  # Drop alpha values.
        rgb_colors = np.asarray(colors * 255, dtype=int)
        return rgb_colors
