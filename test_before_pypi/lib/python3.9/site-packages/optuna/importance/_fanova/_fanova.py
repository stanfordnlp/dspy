"""An implementation of `An Efficient Approach for Assessing Hyperparameter Importance`.

See http://proceedings.mlr.press/v32/hutter14.pdf and https://automl.github.io/fanova/cite.html
for how to cite the original work.

This implementation is inspired by the efficient algorithm in
`fanova` (https://github.com/automl/fanova) and
`pyrfr` (https://github.com/automl/random_forest_run) by the original authors.

Differences include relying on scikit-learn to fit random forests
(`sklearn.ensemble.RandomForestRegressor`) and that it is otherwise written entirely in Python.
This stands in contrast to the original implementation which is partially written in C++.
Since Python runtime overhead may become noticeable, included are instead several
optimizations, e.g. vectorized NumPy functions to compute the marginals, instead of keeping all
running statistics. Known cases include assessing categorical features with a larger
number of choices since each choice is given a unique one-hot encoded raw feature.
"""

from __future__ import annotations

import numpy as np

from optuna._imports import try_import
from optuna.importance._fanova._tree import _FanovaTree


with try_import() as _imports:
    from sklearn.ensemble import RandomForestRegressor


class _Fanova:
    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        min_samples_split: int | float,
        min_samples_leaf: int | float,
        seed: int | None,
    ) -> None:
        _imports.check()

        self._forest = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        self._trees: list[_FanovaTree] | None = None
        self._variances: dict[int, np.ndarray] | None = None
        self._column_to_encoded_columns: list[np.ndarray] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        search_spaces: np.ndarray,
        column_to_encoded_columns: list[np.ndarray],
    ) -> None:
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == search_spaces.shape[0]
        assert search_spaces.shape[1] == 2

        self._forest.fit(X, y)

        self._trees = [_FanovaTree(e.tree_, search_spaces) for e in self._forest.estimators_]
        self._column_to_encoded_columns = column_to_encoded_columns
        self._variances = {}

        if all(tree.variance == 0 for tree in self._trees):
            # If all trees have 0 variance, we cannot assess any importances.
            # This could occur if for instance `X.shape[0] == 1`.
            raise RuntimeError("Encountered zero total variance in all trees.")

    def get_importance(self, feature: int) -> tuple[float, float]:
        # Assert that `fit` has been called.
        assert self._trees is not None
        assert self._variances is not None

        self._compute_variances(feature)

        fractions: list[float] | np.ndarray = []

        for tree_index, tree in enumerate(self._trees):
            tree_variance = tree.variance
            if tree_variance > 0.0:
                fraction = self._variances[feature][tree_index] / tree_variance
                fractions = np.append(fractions, fraction)

        fractions = np.asarray(fractions)

        return float(fractions.mean()), float(fractions.std())

    def _compute_variances(self, feature: int) -> None:
        assert self._trees is not None
        assert self._variances is not None
        assert self._column_to_encoded_columns is not None

        if feature in self._variances:
            return

        raw_features = self._column_to_encoded_columns[feature]
        variances = np.empty(len(self._trees), dtype=np.float64)

        for tree_index, tree in enumerate(self._trees):
            marginal_variance = tree.get_marginal_variance(raw_features)
            variances[tree_index] = np.clip(marginal_variance, 0.0, None)
        self._variances[feature] = variances
