from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _get_trans_params
from optuna.importance._base import _param_importances_to_dict
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova
from optuna.study import Study
from optuna.trial import FrozenTrial


class FanovaImportanceEvaluator(BaseImportanceEvaluator):
    """fANOVA importance evaluator.

    Implements the fANOVA hyperparameter importance evaluation algorithm in
    `An Efficient Approach for Assessing Hyperparameter Importance
    <http://proceedings.mlr.press/v32/hutter14.html>`__.

    fANOVA fits a random forest regression model that predicts the objective values
    of :class:`~optuna.trial.TrialState.COMPLETE` trials given their parameter configurations.
    The more accurate this model is, the more reliable the importances assessed
    by this class are.

    .. note::

        This class takes over 1 minute when given a study that contains 1000+ trials.
        We published `optuna-fast-fanova <https://github.com/optuna/optuna-fast-fanova>`__ library,
        that is a Cython accelerated fANOVA implementation. By using it, you can get hyperparameter
        importances within a few seconds.

    .. note::

        Requires the `sklearn <https://github.com/scikit-learn/scikit-learn>`__ Python package.

    .. note::

        The performance of fANOVA depends on the prediction performance of the underlying
        random forest model. In order to obtain high prediction performance, it is necessary to
        cover a wide range of the hyperparameter search space. It is recommended to use an
        exploration-oriented sampler such as :class:`~optuna.samplers.RandomSampler`.

    .. note::

        For how to cite the original work, please refer to
        https://automl.github.io/fanova/cite.html.

    Args:
        n_trees:
            The number of trees in the forest.
        max_depth:
            The maximum depth of the trees in the forest.
        seed:
            Controls the randomness of the forest. For deterministic behavior, specify a value
            other than :obj:`None`.

    """

    def __init__(self, *, n_trees: int = 64, max_depth: int = 64, seed: int | None = None) -> None:
        self._evaluator = _Fanova(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            seed=seed,
        )

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        distributions = _get_distributions(study, params=params)
        if params is None:
            params = list(distributions.keys())
        assert params is not None

        # fANOVA does not support parameter distributions with a single value.
        # However, there is no reason to calculate parameter importance in such case anyway,
        # since it will always be 0 as the parameter is constant in the objective function.
        non_single_distributions = {
            name: dist for name, dist in distributions.items() if not dist.single()
        }
        single_distributions = {
            name: dist for name, dist in distributions.items() if dist.single()
        }

        if len(non_single_distributions) == 0:
            return {}

        trials: list[FrozenTrial] = _get_filtered_trials(study, params=params, target=target)

        trans = _SearchSpaceTransform(
            non_single_distributions, transform_log=False, transform_step=False
        )

        trans_params: np.ndarray = _get_trans_params(trials, trans)
        target_values: np.ndarray = _get_target_values(trials, target)

        evaluator = self._evaluator
        evaluator.fit(
            X=trans_params,
            y=target_values,
            search_spaces=trans.bounds,
            column_to_encoded_columns=trans.column_to_encoded_columns,
        )
        param_importances = np.array(
            [evaluator.get_importance(i)[0] for i in range(len(non_single_distributions))]
        )

        return _sort_dict_by_importance(
            {
                **_param_importances_to_dict(non_single_distributions.keys(), param_importances),
                **_param_importances_to_dict(single_distributions.keys(), 0.0),
            }
        )
