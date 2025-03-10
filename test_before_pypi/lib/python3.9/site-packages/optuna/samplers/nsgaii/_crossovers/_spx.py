from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class SPXCrossover(BaseCrossover):
    """Simplex Crossover operation used by :class:`~optuna.samplers.NSGAIISampler`.

    Uniformly samples child individuals from within a single simplex
    that is similar to the simplex produced by the parent individual.
    For further information about SPX crossover, please refer to the following paper:

    - `Shigeyoshi Tsutsui and Shigeyoshi Tsutsui and David E. Goldberg and
      David E. Goldberg and Kumara Sastry and Kumara Sastry
      Progress Toward Linkage Learning in Real-Coded GAs with Simplex Crossover.
      IlliGAL Report. 2000.
      <https://www.researchgate.net/publication/2388486_Progress_Toward_Linkage_Learning_in_Real-Coded_GAs_with_Simplex_Crossover>`__

    Args:
        epsilon:
            Expansion rate. If not specified, defaults to ``sqrt(len(search_space) + 2)``.
    """

    n_parents = 3

    def __init__(self, epsilon: float | None = None) -> None:
        self._epsilon = epsilon

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:
        # https://www.researchgate.net/publication/2388486_Progress_Toward_Linkage_Learning_in_Real-Coded_GAs_with_Simplex_Crossover
        # Section 2 A Brief Review of SPX

        n = self.n_parents - 1
        G = np.mean(parents_params, axis=0)  # Equation (1).
        rs = np.power(rng.rand(n), 1 / (np.arange(n) + 1))  # Equation (2).

        epsilon = np.sqrt(len(search_space_bounds) + 2) if self._epsilon is None else self._epsilon
        xks = [G + epsilon * (pk - G) for pk in parents_params]  # Equation (3).

        ck = 0  # Equation (4).
        for k in range(1, self.n_parents):
            ck = rs[k - 1] * (xks[k - 1] - xks[k] + ck)

        child_params = xks[-1] + ck  # Equation (5).

        return child_params
