from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import optuna
from optuna.distributions import BaseDistribution
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go


logger = get_logger(__name__)


class _ImportancesInfo(NamedTuple):
    importance_values: list[float]
    param_names: list[str]
    importance_labels: list[str]
    target_name: str


def _get_importances_info(
    study: Study,
    evaluator: BaseImportanceEvaluator | None,
    params: list[str] | None,
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
) -> _ImportancesInfo:
    _check_plot_args(study, target, target_name)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    if len(trials) == 0:
        logger.warning("Study instance does not contain completed trials.")
        return _ImportancesInfo(
            importance_values=[],
            param_names=[],
            importance_labels=[],
            target_name=target_name,
        )

    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, params=params, target=target
    )

    importances = dict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())
    importance_labels = [f"{val:.2f}" if val >= 0.01 else "<0.01" for val in importance_values]

    return _ImportancesInfo(
        importance_values=importance_values,
        param_names=param_names,
        importance_labels=importance_labels,
        target_name=target_name,
    )


def _get_importances_infos(
    study: Study,
    evaluator: BaseImportanceEvaluator | None,
    params: list[str] | None,
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
) -> tuple[_ImportancesInfo, ...]:
    metric_names = study.metric_names
    if target or not study._is_multi_objective():
        target_name = metric_names[0] if metric_names is not None and not target else target_name
        importances_infos: tuple[_ImportancesInfo, ...] = (
            _get_importances_info(
                study,
                evaluator,
                params,
                target=target,
                target_name=target_name,
            ),
        )

    else:
        n_objectives = len(study.directions)
        target_names = (
            metric_names
            if metric_names is not None
            else (f"{target_name} {objective_id}" for objective_id in range(n_objectives))
        )

        importances_infos = tuple(
            _get_importances_info(
                study,
                evaluator,
                params,
                target=lambda t: t.values[objective_id],
                target_name=target_name,
            )
            for objective_id, target_name in enumerate(target_names)
        )

    return importances_infos


def plot_param_importances(
    study: Study,
    evaluator: BaseImportanceEvaluator | None = None,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot hyperparameter importances.

    .. seealso::

        This function visualizes the results of :func:`optuna.importance.get_param_importances`.

    Args:
        study:
            An optimized study.
        evaluator:
            An importance evaluator object that specifies which algorithm to base the importance
            assessment on.
            Defaults to
            :class:`~optuna.importance.FanovaImportanceEvaluator`.

            .. note::
                :class:`~optuna.importance.FanovaImportanceEvaluator` takes over 1 minute
                when given a study that contains 1000+ trials. We published
                `optuna-fast-fanova <https://github.com/optuna/optuna-fast-fanova>`__ library,
                that is a Cython accelerated fANOVA implementation.
                By using it, you can get hyperparameter importances within a few seconds.

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
            Target's name to display on the legend. Names set via
            :meth:`~optuna.study.Study.set_metric_names` will be used if ``target`` is :obj:`None`,
            overriding this argument.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()
    importances_infos = _get_importances_infos(study, evaluator, params, target, target_name)
    return _get_importances_plot(importances_infos, study)


def _get_importances_plot(infos: tuple[_ImportancesInfo, ...], study: Study) -> "go.Figure":
    layout = go.Layout(
        title="Hyperparameter Importances",
        xaxis={"title": "Hyperparameter Importance"},
        yaxis={"title": "Hyperparameter"},
    )

    data: list[go.Bar] = []
    for info in infos:
        if not info.importance_values:
            continue

        data.append(
            go.Bar(
                x=info.importance_values,
                y=info.param_names,
                name=info.target_name,
                text=info.importance_labels,
                textposition="outside",
                cliponaxis=False,  # Ensure text is not clipped.
                hovertemplate=_get_hover_template(info, study),
                orientation="h",
            )
        )

    return go.Figure(data, layout)


def _get_distribution(param_name: str, study: Study) -> BaseDistribution:
    for trial in study.trials:
        if param_name in trial.distributions:
            return trial.distributions[param_name]
    assert False


def _make_hovertext(param_name: str, importance: float, study: Study) -> str:
    return "{} ({}): {}<extra></extra>".format(
        param_name, _get_distribution(param_name, study).__class__.__name__, importance
    )


def _get_hover_template(importances_info: _ImportancesInfo, study: Study) -> list[str]:
    return [
        _make_hovertext(param_name, importance, study)
        for param_name, importance in zip(
            importances_info.param_names, importances_info.importance_values
        )
    ]
