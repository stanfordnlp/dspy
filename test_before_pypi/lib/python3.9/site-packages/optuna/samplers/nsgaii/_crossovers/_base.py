from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from optuna.study import Study


class BaseCrossover(abc.ABC):
    """Base class for crossovers.

    A crossover operation is used by :class:`~optuna.samplers.NSGAIISampler`
    to create new parameter combination from parameters of ``n`` parent individuals.

    .. note::
        Concrete implementations of this class are expected to only accept parameters
        from numerical distributions. At the moment, only crossover operation for categorical
        parameters (uniform crossover) is built-in into :class:`~optuna.samplers.NSGAIISampler`.
    """

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def n_parents(self) -> int:
        """Number of parent individuals required to perform crossover."""

        raise NotImplementedError

    @abc.abstractmethod
    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:
        """Perform crossover of selected parent individuals.

        This method is called in :func:`~optuna.samplers.NSGAIISampler.sample_relative`.

        Args:
            parents_params:
                A ``numpy.ndarray`` with dimensions ``num_parents x num_parameters``.
                Represents a parameter space for each parent individual. This space is
                continuous for numerical parameters.
            rng:
                An instance of ``numpy.random.RandomState``.
            study:
                Target study object.
            search_space_bounds:
                A ``numpy.ndarray`` with dimensions ``len_search_space x 2`` representing
                numerical distribution bounds constructed from transformed search space.

        Returns:
            A 1-dimensional ``numpy.ndarray`` containing new parameter combination.
        """

        raise NotImplementedError
