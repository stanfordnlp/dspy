from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _get_trans_params
from optuna.importance._base import _param_importances_to_dict
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial


with try_import() as _imports:
    from sklearn.ensemble import RandomForestRegressor


class MeanDecreaseImpurityImportanceEvaluator(BaseImportanceEvaluator):
    """Mean Decrease Impurity (MDI) parameter importance evaluator.

    This evaluator fits fits a random forest regression model that predicts the objective values
    of :class:`~optuna.trial.TrialState.COMPLETE` trials given their parameter configurations.
    Feature importances are then computed using MDI.

    .. note::

        This evaluator requires the `sklearn <https://scikit-learn.org/stable/>`__ Python package
        and is based on `sklearn.ensemble.RandomForestClassifier.feature_importances_
        <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_>`__.

    Args:
        n_trees:
            Number of trees in the random forest.
        max_depth:
            The maximum depth of each tree in the random forest.
        seed:
            Seed for the random forest.
    """

    def __init__(self, *, n_trees: int = 64, max_depth: int = 64, seed: int | None = None) -> None:
        _imports.check()

        self._forest = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
        )
        self._trans_params = np.empty(0)
        self._trans_values = np.empty(0)
        self._param_names: list[str] = list()

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
        if len(params) == 0:
            return {}

        trials: list[FrozenTrial] = _get_filtered_trials(study, params=params, target=target)
        trans = _SearchSpaceTransform(distributions, transform_log=False, transform_step=False)
        trans_params: np.ndarray = _get_trans_params(trials, trans)
        target_values: np.ndarray = _get_target_values(trials, target)

        forest = self._forest
        forest.fit(X=trans_params, y=target_values)
        feature_importances = forest.feature_importances_

        # Untransform feature importances to param importances
        # by adding up relevant feature importances.
        param_importances = np.zeros(len(params))
        np.add.at(param_importances, trans.encoded_column_to_column, feature_importances)

        return _sort_dict_by_importance(_param_importances_to_dict(params, param_importances))
