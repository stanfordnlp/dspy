from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._edf import _get_edf_info
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental_func("2.2.0")
def plot_edf(
    study: Study | Sequence[Study],
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the objective value EDF (empirical distribution function) of a study with Matplotlib.

    Note that only the complete trials are considered when plotting the EDF.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_edf` for an example,
        where this function can be replaced with it.

    .. note::

        Please refer to `matplotlib.pyplot.legend
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_
        to adjust the style of the generated legend.

    Args:
        study:
            A target :class:`~optuna.study.Study` object.
            You can pass multiple studies if you want to compare those EDFs.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Empirical Distribution Function Plot")
    ax.set_xlabel(target_name)
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1)
    cmap = plt.get_cmap("tab20")  # Use tab20 colormap for multiple line plots.

    info = _get_edf_info(study, target, target_name)
    edf_lines = info.lines

    if len(edf_lines) == 0:
        return ax

    for i, (study_name, y_values) in enumerate(edf_lines):
        ax.plot(info.x_values, y_values, color=cmap(i), alpha=0.7, label=study_name)

    if len(edf_lines) >= 2:
        ax.legend()

    return ax
