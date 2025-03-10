from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._pareto_front import _get_pareto_front_info
from optuna.visualization._pareto_front import _ParetoFrontInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("2.8.0")
def plot_pareto_front(
    study: Study,
    *,
    target_names: list[str] | None = None,
    include_dominated_trials: bool = True,
    axis_order: list[int] | None = None,
    constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    targets: Callable[[FrozenTrial], Sequence[float]] | None = None,
) -> "Axes":
    """Plot the Pareto front of a study.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_pareto_front` for an example.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values. ``study.n_objectives`` must be either 2 or 3 when ``targets`` is :obj:`None`.
        target_names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead. If ``targets`` is specified
            for a study that does not contain any completed trial,
            ``target_name`` must be specified.
        include_dominated_trials:
            A flag to include all dominated trial's objective values.
        axis_order:
            A list of indices indicating the axis order. If :obj:`None` is specified,
            default order is used. ``axis_order`` and ``targets`` cannot be used at the same time.

            .. warning::
                Deprecated in v3.0.0. This feature will be removed in the future. The removal of
                this feature is currently scheduled for v5.0.0, but this schedule is subject to
                change. See https://github.com/optuna/optuna/releases/tag/v3.0.0.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraint is violated. A value equal to or smaller than 0 is considered feasible.
            This specification is the same as in, for example,
            :class:`~optuna.samplers.NSGAIISampler`.

            If given, trials are classified into three categories: feasible and best, feasible but
            non-best, and infeasible. Categories are shown in different colors. Here, whether a
            trial is best (on Pareto front) or not is determined ignoring all infeasible trials.

            .. warning::
                Deprecated in v4.0.0. This feature will be removed in the future. The removal of
                this feature is currently scheduled for v6.0.0, but this schedule is subject to
                change. See https://github.com/optuna/optuna/releases/tag/v4.0.0.
        targets:
            A function that returns a tuple of target values to display.
            The argument to this function is :class:`~optuna.trial.FrozenTrial`.
            ``targets`` must be :obj:`None` or return 2 or 3 values.
            ``axis_order`` and ``targets`` cannot be used at the same time.
            If the number of objectives is neither 2 nor 3, ``targets`` must be specified.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.0.0.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()

    info = _get_pareto_front_info(
        study, target_names, include_dominated_trials, axis_order, constraints_func, targets
    )
    return _get_pareto_front_plot(info)


def _get_pareto_front_plot(info: _ParetoFrontInfo) -> "Axes":
    if info.n_targets == 2:
        return _get_pareto_front_2d(info)
    elif info.n_targets == 3:
        return _get_pareto_front_3d(info)
    else:
        assert False, "Must not reach here"


def _get_pareto_front_2d(info: _ParetoFrontInfo) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.set_xlabel(info.target_names[info.axis_order[0]])
    ax.set_ylabel(info.target_names[info.axis_order[1]])

    trial_label: str = "Trial"
    if len(info.infeasible_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.infeasible_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.infeasible_trials_with_values],
            color="#cccccc",
            label="Infeasible Trial",
        )
        trial_label = "Feasible Trial"
    if len(info.non_best_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.non_best_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.non_best_trials_with_values],
            color=cmap(0),
            label=trial_label,
        )
    if len(info.best_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.best_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.best_trials_with_values],
            color=cmap(3),
            label="Best Trial",
        )

    if info.non_best_trials_with_values is not None and ax.has_data():
        ax.legend()

    return ax


def _get_pareto_front_3d(info: _ParetoFrontInfo) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.set_xlabel(info.target_names[info.axis_order[0]])
    ax.set_ylabel(info.target_names[info.axis_order[1]])
    ax.set_zlabel(info.target_names[info.axis_order[2]])

    trial_label: str = "Trial"
    if (
        info.infeasible_trials_with_values is not None
        and len(info.infeasible_trials_with_values) > 0
    ):
        ax.scatter(
            xs=[values[info.axis_order[0]] for _, values in info.infeasible_trials_with_values],
            ys=[values[info.axis_order[1]] for _, values in info.infeasible_trials_with_values],
            zs=[values[info.axis_order[2]] for _, values in info.infeasible_trials_with_values],
            color="#cccccc",
            label="Infeasible Trial",
        )
        trial_label = "Feasible Trial"

    if info.non_best_trials_with_values is not None and len(info.non_best_trials_with_values) > 0:
        ax.scatter(
            xs=[values[info.axis_order[0]] for _, values in info.non_best_trials_with_values],
            ys=[values[info.axis_order[1]] for _, values in info.non_best_trials_with_values],
            zs=[values[info.axis_order[2]] for _, values in info.non_best_trials_with_values],
            color=cmap(0),
            label=trial_label,
        )

    if info.best_trials_with_values is not None and len(info.best_trials_with_values):
        ax.scatter(
            xs=[values[info.axis_order[0]] for _, values in info.best_trials_with_values],
            ys=[values[info.axis_order[1]] for _, values in info.best_trials_with_values],
            zs=[values[info.axis_order[2]] for _, values in info.best_trials_with_values],
            color=cmap(3),
            label="Best Trial",
        )

    if info.non_best_trials_with_values is not None and ax.has_data():
        ax.legend()

    return ax
