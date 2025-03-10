from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna._experimental import experimental_func
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._param_importances import _get_importances_infos
from optuna.visualization._param_importances import _ImportancesInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Figure
    from optuna.visualization.matplotlib._matplotlib_imports import plt


_logger = get_logger(__name__)


AXES_PADDING_RATIO = 1.05


@experimental_func("2.2.0")
def plot_param_importances(
    study: Study,
    evaluator: BaseImportanceEvaluator | None = None,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot hyperparameter importances with Matplotlib.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_param_importances` for an example.

    Args:
        study:
            An optimized study.
        evaluator:
            An importance evaluator object that specifies which algorithm to base the importance
            assessment on.
            Defaults to
            :class:`~optuna.importance.FanovaImportanceEvaluator`.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.
            For multi-objective optimization, all objectives will be plotted if ``target``
            is :obj:`None`.

            .. note::
                This argument can be used to specify which objective to plot if ``study`` is being
                used for multi-objective optimization. For example, to get only the hyperparameter
                importance of the first objective, use ``target=lambda t: t.values[0]`` for the
                target parameter.
        target_name:
            Target's name to display on the axis label. Names set via
            :meth:`~optuna.study.Study.set_metric_names` will be used if ``target`` is :obj:`None`,
            overriding this argument.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()
    importances_infos = _get_importances_infos(study, evaluator, params, target, target_name)
    return _get_importances_plot(importances_infos)


def _get_importances_plot(infos: tuple[_ImportancesInfo, ...]) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Hyperparameter Importances", loc="left")
    ax.set_xlabel("Hyperparameter Importance")
    ax.set_ylabel("Hyperparameter")
    height = 0.8 / len(infos)  # Default height split between objectives.

    for objective_id, info in enumerate(infos):
        param_names = info.param_names
        pos = np.arange(len(param_names))
        offset = height * objective_id
        importance_values = info.importance_values

        if not importance_values:
            continue

        # Draw horizontal bars.
        ax.barh(
            pos + offset,
            importance_values,
            height=height,
            align="center",
            label=info.target_name,
            color=plt.get_cmap("tab20c")(objective_id),
        )

        _set_bar_labels(info, fig, ax, offset)
        ax.set_yticks(pos + offset / 2, param_names)

    ax.legend(loc="best")
    return ax


def _set_bar_labels(info: _ImportancesInfo, fig: "Figure", ax: "Axes", offset: float) -> None:
    renderer = fig.canvas.get_renderer()
    for idx, (val, label) in enumerate(zip(info.importance_values, info.importance_labels)):
        text = ax.text(val, idx + offset, label, va="center")

        # Sometimes horizontal axis needs to be re-scaled
        # to avoid text going over plot area.
        bbox = text.get_window_extent(renderer)
        bbox = bbox.transformed(ax.transData.inverted())
        _, plot_xmax = ax.get_xlim()
        bbox_xmax = bbox.xmax

        if bbox_xmax > plot_xmax:
            ax.set_xlim(xmax=AXES_PADDING_RATIO * bbox_xmax)
