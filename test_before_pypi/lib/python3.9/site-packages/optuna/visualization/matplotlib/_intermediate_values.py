from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.visualization._intermediate_values import _get_intermediate_plot_info
from optuna.visualization._intermediate_values import _IntermediatePlotInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental_func("2.2.0")
def plot_intermediate_values(study: Study) -> "Axes":
    """Plot intermediate values of all trials in a study with Matplotlib.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_intermediate_values` for an example.

    .. note::
        Please refer to `matplotlib.pyplot.legend
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`__
        to adjust the style of the generated legend.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()
    return _get_intermediate_plot(_get_intermediate_plot_info(study))


def _get_intermediate_plot(info: _IntermediatePlotInfo) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots(tight_layout=True)
    ax.set_title("Intermediate Values Plot")
    ax.set_xlabel("Step")
    ax.set_ylabel("Intermediate Value")
    cmap = plt.get_cmap("tab20")  # Use tab20 colormap for multiple line plots.

    trial_infos = info.trial_infos

    for i, tinfo in enumerate(trial_infos):
        ax.plot(
            tuple((x for x, _ in tinfo.sorted_intermediate_values)),
            tuple((y for _, y in tinfo.sorted_intermediate_values)),
            color=cmap(i) if tinfo.feasible else "#CCCCCC",
            marker=".",
            alpha=0.7,
            label="Trial{}".format(tinfo.trial_number),
        )

    if len(trial_infos) >= 2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    return ax
