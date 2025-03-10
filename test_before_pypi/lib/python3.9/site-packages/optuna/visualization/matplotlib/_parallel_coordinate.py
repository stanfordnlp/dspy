from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._parallel_coordinate import _get_parallel_coordinate_info
from optuna.visualization._parallel_coordinate import _ParallelCoordinateInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import LineCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("2.2.0")
def plot_parallel_coordinate(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the high-dimensional parameter relationships in a study with Matplotlib.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_parallel_coordinate` for an example.

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
            Target's name to display on the axis label and the legend.

    Returns:
        A :class:`matplotlib.axes.Axes` object.

    .. note::
        The colormap is reversed when the ``target`` argument isn't :obj:`None` or ``direction``
        of :class:`~optuna.study.Study` is ``minimize``.
    """

    _imports.check()
    info = _get_parallel_coordinate_info(study, params, target, target_name)
    return _get_parallel_coordinate_plot(info)


def _get_parallel_coordinate_plot(info: _ParallelCoordinateInfo) -> "Axes":
    reversescale = info.reverse_scale
    target_name = info.target_name

    # Set up the graph style.
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Blues_r" if reversescale else "Blues")
    ax.set_title("Parallel Coordinate Plot")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Prepare data for plotting.
    if len(info.dims_params) == 0 or len(info.dim_objective.values) == 0:
        return ax

    obj_min = info.dim_objective.range[0]
    obj_max = info.dim_objective.range[1]
    obj_w = obj_max - obj_min
    dims_obj_base = [[o] for o in info.dim_objective.values]
    for dim in info.dims_params:
        p_min = dim.range[0]
        p_max = dim.range[1]
        p_w = p_max - p_min

        if p_w == 0.0:
            center = obj_w / 2 + obj_min
            for i in range(len(dim.values)):
                dims_obj_base[i].append(center)
        else:
            for i, v in enumerate(dim.values):
                dims_obj_base[i].append((v - p_min) / p_w * obj_w + obj_min)

    # Draw multiple line plots and axes.
    # Ref: https://stackoverflow.com/a/50029441
    n_params = len(info.dims_params)
    ax.set_xlim(0, n_params)
    ax.set_ylim(info.dim_objective.range[0], info.dim_objective.range[1])
    xs = [range(n_params + 1) for _ in range(len(dims_obj_base))]
    segments = [np.column_stack([x, y]) for x, y in zip(xs, dims_obj_base)]
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(np.asarray(info.dim_objective.values))
    axcb = fig.colorbar(lc, pad=0.1, ax=ax)
    axcb.set_label(target_name)
    var_names = [info.dim_objective.label] + [dim.label for dim in info.dims_params]
    plt.xticks(range(n_params + 1), var_names, rotation=330)

    for i, dim in enumerate(info.dims_params):
        ax2 = ax.twinx()
        if dim.is_log:
            ax2.set_ylim(np.power(10, dim.range[0]), np.power(10, dim.range[1]))
            ax2.set_yscale("log")
        else:
            ax2.set_ylim(dim.range[0], dim.range[1])
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.spines["right"].set_position(("axes", (i + 1) / n_params))
        if dim.is_cat:
            ax2.set_yticks(dim.tickvals)
            ax2.set_yticklabels(dim.ticktext)

    ax.add_collection(lc)

    return ax
