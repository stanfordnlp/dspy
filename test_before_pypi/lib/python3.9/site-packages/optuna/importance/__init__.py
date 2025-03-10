from __future__ import annotations

from collections.abc import Callable

from optuna._experimental import warn_experimental_argument
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova import FanovaImportanceEvaluator
from optuna.importance._mean_decrease_impurity import MeanDecreaseImpurityImportanceEvaluator
from optuna.importance._ped_anova import PedAnovaImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial


__all__ = [
    "BaseImportanceEvaluator",
    "FanovaImportanceEvaluator",
    "MeanDecreaseImpurityImportanceEvaluator",
    "PedAnovaImportanceEvaluator",
    "get_param_importances",
]


def get_param_importances(
    study: Study,
    *,
    evaluator: BaseImportanceEvaluator | None = None,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Evaluate parameter importances based on completed trials in the given study.

    The parameter importances are returned as a dictionary where the keys consist of parameter
    names and their values importances.
    The importances are represented by non-negative floating point numbers, where higher values
    mean that the parameters are more important.
    The returned dictionary is ordered by its values in a descending order.
    By default, the sum of the importance values are normalized to 1.0.

    If ``params`` is :obj:`None`, all parameter that are present in all of the completed trials are
    assessed.
    This implies that conditional parameters will be excluded from the evaluation.
    To assess the importances of conditional parameters, a :obj:`list` of parameter names can be
    specified via ``params``.
    If specified, only completed trials that contain all of the parameters will be considered.
    If no such trials are found, an error will be raised.

    If the given study does not contain completed trials, an error will be raised.

    .. note::

        If ``params`` is specified as an empty list, an empty dictionary is returned.

    .. seealso::

        See :func:`~optuna.visualization.plot_param_importances` to plot importances.

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
                If ``n_trials`` is more than 10000, the Cython implementation takes more than
                a minute, so you can use :class:`~optuna.importance.PedAnovaImportanceEvaluator`
                instead, enabling the evaluation to finish in a second.

        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.
        target:
            A function to specify the value to evaluate importances.
            If it is :obj:`None` and ``study`` is being used for single-objective optimization,
            the objective values are used. ``target`` must be specified if ``study`` is being
            used for multi-objective optimization.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective
                optimization. For example, to get the hyperparameter importance of the first
                objective, use ``target=lambda t: t.values[0]`` for the target parameter.
        normalize:
            A boolean option to specify whether the sum of the importance values should be
            normalized to 1.0.
            Defaults to :obj:`True`.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v3.0.0.

    Returns:
        A :obj:`dict` where the keys are parameter names and the values are assessed importances.

    """
    if evaluator is None:
        evaluator = FanovaImportanceEvaluator()

    if not isinstance(evaluator, BaseImportanceEvaluator):
        raise TypeError("Evaluator must be a subclass of BaseImportanceEvaluator.")

    res = evaluator.evaluate(study, params=params, target=target)
    if normalize:
        s = sum(res.values())
        if s == 0.0:
            n_params = len(res)
            return dict((param, 1.0 / n_params) for param in res.keys())
        else:
            return dict((param, value / s) for (param, value) in res.items())
    else:
        warn_experimental_argument("normalize")
        return res
