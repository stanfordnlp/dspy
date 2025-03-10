from __future__ import annotations

from functools import lru_cache
import itertools
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import sklearn.tree


class _FanovaTree:
    def __init__(self, tree: "sklearn.tree._tree.Tree", search_spaces: np.ndarray) -> None:
        assert search_spaces.shape[0] == tree.n_features
        assert search_spaces.shape[1] == 2

        self._tree = tree
        self._search_spaces = search_spaces

        statistics = self._precompute_statistics()
        split_midpoints, split_sizes = self._precompute_split_midpoints_and_sizes()
        subtree_active_features = self._precompute_subtree_active_features()

        self._statistics = statistics
        self._split_midpoints = split_midpoints
        self._split_sizes = split_sizes
        self._subtree_active_features = subtree_active_features
        self._variance = None  # Computed lazily and requires `self._statistics`.

    @property
    def variance(self) -> float:
        if self._variance is None:
            leaf_node_indices = np.nonzero(np.array(self._tree.feature) < 0)[0]
            statistics = self._statistics[leaf_node_indices]
            values = statistics[:, 0]
            weights = statistics[:, 1]
            average_values = np.average(values, weights=weights)
            variance = np.average((values - average_values) ** 2, weights=weights)

            self._variance = variance

        assert self._variance is not None
        return self._variance

    def get_marginal_variance(self, features: np.ndarray) -> float:
        assert features.size > 0

        # For each midpoint along the given dimensions, traverse this tree to compute the
        # marginal predictions.
        selected_midpoints = [self._split_midpoints[f] for f in features]
        selected_sizes = [self._split_sizes[f] for f in features]

        product_midpoints = itertools.product(*selected_midpoints)
        product_sizes = itertools.product(*selected_sizes)

        sample = np.full(self._n_features, fill_value=np.nan, dtype=np.float64)

        values: list[float] | np.ndarray = []
        weights: list[float] | np.ndarray = []

        for midpoints, sizes in zip(product_midpoints, product_sizes):
            sample[features] = np.array(midpoints)

            value, weight = self._get_marginalized_statistics(sample)
            weight *= float(np.prod(sizes))

            values = np.append(values, value)
            weights = np.append(weights, weight)

        weights = np.asarray(weights)
        values = np.asarray(values)
        average_values = np.average(values, weights=weights)
        variance = np.average((values - average_values) ** 2, weights=weights)

        assert variance >= 0.0
        return variance

    def _get_marginalized_statistics(self, feature_vector: np.ndarray) -> tuple[float, float]:
        assert feature_vector.size == self._n_features

        marginalized_features = np.isnan(feature_vector)
        active_features = ~marginalized_features

        # Reduce search space cardinalities to 1 for non-active features.
        search_spaces = self._search_spaces.copy()
        search_spaces[marginalized_features] = [0.0, 1.0]

        # Start from the root and traverse towards the leafs.
        active_nodes = [0]
        active_search_spaces = [search_spaces]

        node_indices = []
        active_leaf_search_spaces = []

        while len(active_nodes) > 0:
            node_index = active_nodes.pop()
            search_spaces = active_search_spaces.pop()

            feature = self._get_node_split_feature(node_index)
            if feature >= 0:  # Not leaf. Avoid unnecessary call to `_is_node_leaf`.
                # If node splits on an active feature, push the child node that we end up in.
                response = feature_vector[feature]
                if not np.isnan(response):
                    if response <= self._get_node_split_threshold(node_index):
                        next_node_index = self._get_node_left_child(node_index)
                        next_subspace = self._get_node_left_child_subspaces(
                            node_index, search_spaces
                        )
                    else:
                        next_node_index = self._get_node_right_child(node_index)
                        next_subspace = self._get_node_right_child_subspaces(
                            node_index, search_spaces
                        )

                    active_nodes.append(next_node_index)
                    active_search_spaces.append(next_subspace)
                    continue

                # If subtree starting from node splits on an active feature, push both child nodes.
                # Here, we use `any` for list because `ndarray.any` is slow.
                if any(self._subtree_active_features[node_index][active_features].tolist()):
                    for child_node_index in self._get_node_children(node_index):
                        active_nodes.append(child_node_index)
                        active_search_spaces.append(search_spaces)
                    continue

            # If node is a leaf or the subtree does not split on any of the active features.
            node_indices.append(node_index)
            active_leaf_search_spaces.append(search_spaces)

        statistics = self._statistics[node_indices]
        values = statistics[:, 0]
        weights = statistics[:, 1]
        active_features_cardinalities = _get_cardinality_batched(active_leaf_search_spaces)
        weights = weights / active_features_cardinalities

        value = np.average(values, weights=weights)
        weight = weights.sum()

        return value, weight

    def _precompute_statistics(self) -> np.ndarray:
        n_nodes = self._n_nodes

        # Holds for each node, its weighted average value and the sum of weights.
        statistics = np.empty((n_nodes, 2), dtype=np.float64)

        subspaces = np.array([None for _ in range(n_nodes)])
        subspaces[0] = self._search_spaces

        # Compute marginals for leaf nodes.
        for node_index in range(n_nodes):
            subspace = subspaces[node_index]

            if self._is_node_leaf(node_index):
                value = self._get_node_value(node_index)
                weight = _get_cardinality(subspace)
                statistics[node_index] = [value, weight]
            else:
                for child_node_index, child_subspace in zip(
                    self._get_node_children(node_index),
                    self._get_node_children_subspaces(node_index, subspace),
                ):
                    assert subspaces[child_node_index] is None
                    subspaces[child_node_index] = child_subspace

        # Compute marginals for internal nodes.
        for node_index in reversed(range(n_nodes)):
            if not self._is_node_leaf(node_index):
                child_values = []
                child_weights = []
                for child_node_index in self._get_node_children(node_index):
                    child_values.append(statistics[child_node_index, 0])
                    child_weights.append(statistics[child_node_index, 1])
                value = np.average(child_values, weights=child_weights)
                weight = float(np.sum(child_weights))
                statistics[node_index] = [value, weight]

        return statistics

    def _precompute_split_midpoints_and_sizes(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        midpoints = []
        sizes = []

        search_spaces = self._search_spaces
        for feature, feature_split_values in enumerate(self._compute_features_split_values()):
            feature_split_values = np.concatenate(
                (
                    np.atleast_1d(search_spaces[feature, 0]),
                    feature_split_values,
                    np.atleast_1d(search_spaces[feature, 1]),
                )
            )
            midpoint = 0.5 * (feature_split_values[1:] + feature_split_values[:-1])
            size = feature_split_values[1:] - feature_split_values[:-1]

            midpoints.append(midpoint)
            sizes.append(size)

        return midpoints, sizes

    def _compute_features_split_values(self) -> list[np.ndarray]:
        all_split_values: list[set[float]] = [set() for _ in range(self._n_features)]

        for node_index in range(self._n_nodes):
            feature = self._get_node_split_feature(node_index)
            if feature >= 0:  # Not leaf. Avoid unnecessary call to `_is_node_leaf`.
                threshold = self._get_node_split_threshold(node_index)
                all_split_values[feature].add(threshold)

        sorted_all_split_values: list[np.ndarray] = []

        for split_values in all_split_values:
            split_values_array = np.array(list(split_values), dtype=np.float64)
            split_values_array.sort()
            sorted_all_split_values.append(split_values_array)

        return sorted_all_split_values

    def _precompute_subtree_active_features(self) -> np.ndarray:
        subtree_active_features = np.full((self._n_nodes, self._n_features), fill_value=False)

        for node_index in reversed(range(self._n_nodes)):
            feature = self._get_node_split_feature(node_index)
            if feature >= 0:  # Not leaf. Avoid unnecessary call to `_is_node_leaf`.
                subtree_active_features[node_index, feature] = True
                for child_node_index in self._get_node_children(node_index):
                    subtree_active_features[node_index] |= subtree_active_features[
                        child_node_index
                    ]

        return subtree_active_features

    @property
    def _n_features(self) -> int:
        return len(self._search_spaces)

    @property
    def _n_nodes(self) -> int:
        return self._tree.node_count

    @lru_cache(maxsize=None)
    def _is_node_leaf(self, node_index: int) -> bool:
        return self._tree.feature[node_index] < 0

    @lru_cache(maxsize=None)
    def _get_node_left_child(self, node_index: int) -> int:
        return self._tree.children_left[node_index]

    @lru_cache(maxsize=None)
    def _get_node_right_child(self, node_index: int) -> int:
        return self._tree.children_right[node_index]

    @lru_cache(maxsize=None)
    def _get_node_children(self, node_index: int) -> tuple[int, int]:
        return self._get_node_left_child(node_index), self._get_node_right_child(node_index)

    @lru_cache(maxsize=None)
    def _get_node_value(self, node_index: int) -> float:
        # self._tree.value: sklearn.tree._tree.Tree.value has
        # the shape (node_count, n_outputs, max_n_classes)
        return float(self._tree.value[node_index].reshape(-1)[0])

    @lru_cache(maxsize=None)
    def _get_node_split_threshold(self, node_index: int) -> float:
        return self._tree.threshold[node_index]

    @lru_cache(maxsize=None)
    def _get_node_split_feature(self, node_index: int) -> int:
        return self._tree.feature[node_index]

    def _get_node_left_child_subspaces(
        self, node_index: int, search_spaces: np.ndarray
    ) -> np.ndarray:
        return _get_subspaces(
            search_spaces,
            search_spaces_column=1,
            feature=self._get_node_split_feature(node_index),
            threshold=self._get_node_split_threshold(node_index),
        )

    def _get_node_right_child_subspaces(
        self, node_index: int, search_spaces: np.ndarray
    ) -> np.ndarray:
        return _get_subspaces(
            search_spaces,
            search_spaces_column=0,
            feature=self._get_node_split_feature(node_index),
            threshold=self._get_node_split_threshold(node_index),
        )

    def _get_node_children_subspaces(
        self, node_index: int, search_spaces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._get_node_left_child_subspaces(node_index, search_spaces),
            self._get_node_right_child_subspaces(node_index, search_spaces),
        )


def _get_cardinality(search_spaces: np.ndarray) -> float:
    return np.prod(search_spaces[:, 1] - search_spaces[:, 0])


def _get_cardinality_batched(search_spaces_list: list[np.ndarray]) -> float:
    search_spaces = np.asarray(search_spaces_list)
    return np.prod(search_spaces[:, :, 1] - search_spaces[:, :, 0], axis=1)


def _get_subspaces(
    search_spaces: np.ndarray, *, search_spaces_column: int, feature: int, threshold: float
) -> np.ndarray:
    search_spaces_subspace = np.copy(search_spaces)
    search_spaces_subspace[feature, search_spaces_column] = threshold
    return search_spaces_subspace
