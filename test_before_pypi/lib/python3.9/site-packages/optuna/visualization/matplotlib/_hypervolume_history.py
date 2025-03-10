from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info
from optuna.visualization._hypervolume_history import _HypervolumeHistoryInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("3.3.0")
def plot_hypervolume_history(
    study: Study,
    reference_point: Sequence[float],
) -> "Axes":
    """Plot hypervolume history of all trials in a study with Matplotlib.

    .. note::
        You need to adjust the size of the plot by yourself using ``plt.tight_layout()`` or
        ``plt.savefig(IMAGE_NAME, bbox_inches='tight')``.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their hypervolumes.
            The number of objectives must be 2 or more.

        reference_point:
            A reference point to use for hypervolume computation.
            The dimension of the reference point must be the same as the number of objectives.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()

    if not study._is_multi_objective():
        raise ValueError(
            "Study must be multi-objective. For single-objective optimization, "
            "please use plot_optimization_history instead."
        )

    if len(reference_point) != len(study.directions):
        raise ValueError(
            "The dimension of the reference point must be the same as the number of objectives."
        )

    info = _get_hypervolume_history_info(study, np.asarray(reference_point, dtype=np.float64))
    return _get_hypervolume_history_plot(info)


def _get_hypervolume_history_plot(
    info: _HypervolumeHistoryInfo,
) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Hypervolume History Plot")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Hypervolume")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.plot(
        info.trial_numbers,
        info.values,
        marker="o",
        color=cmap(0),
        alpha=0.5,
    )
    return ax
