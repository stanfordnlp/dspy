from __future__ import annotations

from typing import NamedTuple

import tqdm

import optuna
from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study.study import Study
from optuna.terminator import BaseErrorEvaluator
from optuna.terminator import BaseImprovementEvaluator
from optuna.terminator import CrossValidationErrorEvaluator
from optuna.terminator import RegretBoundEvaluator
from optuna.terminator.erroreval import StaticErrorEvaluator
from optuna.terminator.improvement.evaluator import BestValueStagnationEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


PADDING_RATIO_Y = 0.05
OPACITY = 0.25


class _ImprovementInfo(NamedTuple):
    trial_numbers: list[int]
    improvements: list[float]
    errors: list[float] | None


@experimental_func("3.2.0")
def plot_terminator_improvement(
    study: Study,
    plot_error: bool = False,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
) -> "go.Figure":
    """Plot the potentials for future objective improvement.

    This function visualizes the objective improvement potentials, evaluated
    with ``improvement_evaluator``.
    It helps to determine whether we should continue the optimization or not.
    You can also plot the error evaluated with
    ``error_evaluator`` if the ``plot_error`` argument is set to :obj:`True`.
    Note that this function may take some time to compute
    the improvement potentials.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted
            for their improvement.
        plot_error:
            A flag to show the error. If it is set to :obj:`True`, errors
            evaluated by ``error_evaluator`` are also plotted as line graph.
            Defaults to :obj:`False`.
        improvement_evaluator:
            An object that evaluates the improvement of the objective function.
            Defaults to :class:`~optuna.terminator.RegretBoundEvaluator`.
        error_evaluator:
            An object that evaluates the error inherent in the objective function.
            Defaults to :class:`~optuna.terminator.CrossValidationErrorEvaluator`.
        min_n_trials:
            The minimum number of trials before termination is considered.
            Terminator improvements for trials below this value are
            shown in a lighter color. Defaults to ``20``.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """
    _imports.check()

    info = _get_improvement_info(study, plot_error, improvement_evaluator, error_evaluator)
    return _get_improvement_plot(info, min_n_trials)


def _get_improvement_info(
    study: Study,
    get_error: bool = False,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
) -> _ImprovementInfo:
    if study._is_multi_objective():
        raise ValueError("This function does not support multi-objective optimization study.")

    if improvement_evaluator is None:
        improvement_evaluator = RegretBoundEvaluator()
    if error_evaluator is None:
        if isinstance(improvement_evaluator, BestValueStagnationEvaluator):
            error_evaluator = StaticErrorEvaluator(constant=0)
        else:
            error_evaluator = CrossValidationErrorEvaluator()

    trial_numbers = []
    completed_trials = []
    improvements = []
    errors = []

    for trial in tqdm.tqdm(study.trials):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed_trials.append(trial)

        if len(completed_trials) == 0:
            continue

        trial_numbers.append(trial.number)

        improvement = improvement_evaluator.evaluate(
            trials=completed_trials, study_direction=study.direction
        )
        improvements.append(improvement)

        if get_error:
            error = error_evaluator.evaluate(
                trials=completed_trials, study_direction=study.direction
            )
            errors.append(error)

    if len(errors) == 0:
        return _ImprovementInfo(
            trial_numbers=trial_numbers, improvements=improvements, errors=None
        )
    else:
        return _ImprovementInfo(
            trial_numbers=trial_numbers, improvements=improvements, errors=errors
        )


def _get_improvement_scatter(
    trial_numbers: list[int],
    improvements: list[float],
    opacity: float = 1.0,
    showlegend: bool = True,
) -> "go.Scatter":
    plotly_blue_with_opacity = f"rgba(99, 110, 250, {opacity})"
    return go.Scatter(
        x=trial_numbers,
        y=improvements,
        mode="markers+lines",
        marker=dict(color=plotly_blue_with_opacity),
        line=dict(color=plotly_blue_with_opacity),
        name="Terminator Improvement",
        showlegend=showlegend,
        legendgroup="improvement",
    )


def _get_error_scatter(
    trial_numbers: list[int],
    errors: list[float] | None,
) -> "go.Scatter":
    if errors is None:
        return go.Scatter()

    plotly_red = "rgb(239, 85, 59)"
    return go.Scatter(
        x=trial_numbers,
        y=errors,
        mode="markers+lines",
        name="Error",
        marker=dict(color=plotly_red),
        line=dict(color=plotly_red),
    )


def _get_y_range(info: _ImprovementInfo, min_n_trials: int) -> tuple[float, float]:
    min_value = min(info.improvements)
    if info.errors is not None:
        min_value = min(min_value, min(info.errors))

    # Determine the display range based on trials after min_n_trials.
    if len(info.trial_numbers) > min_n_trials:
        max_value = max(info.improvements[min_n_trials:])
    # If there are no trials after min_trials, determine the display range based on all trials.
    else:
        max_value = max(info.improvements)

    if info.errors is not None:
        max_value = max(max_value, max(info.errors))

    padding = (max_value - min_value) * PADDING_RATIO_Y
    return (min_value - padding, max_value + padding)


def _get_improvement_plot(info: _ImprovementInfo, min_n_trials: int) -> "go.Figure":
    n_trials = len(info.trial_numbers)

    fig = go.Figure(
        layout=go.Layout(
            title="Terminator Improvement Plot",
            xaxis=dict(title="Trial"),
            yaxis=dict(title="Terminator Improvement"),
        )
    )
    if n_trials == 0:
        _logger.warning("There are no complete trials.")
        return fig

    fig.add_trace(
        _get_improvement_scatter(
            info.trial_numbers[: min_n_trials + 1],
            info.improvements[: min_n_trials + 1],
            # Plot line with a lighter color until the number of trials reaches min_n_trials.
            OPACITY,
            n_trials <= min_n_trials,  # Avoid showing legend twice.
        )
    )

    if n_trials > min_n_trials:
        fig.add_trace(
            _get_improvement_scatter(
                info.trial_numbers[min_n_trials:],
                info.improvements[min_n_trials:],
            )
        )

    fig.add_trace(_get_error_scatter(info.trial_numbers, info.errors))

    fig.update_yaxes(range=_get_y_range(info, min_n_trials))
    return fig
