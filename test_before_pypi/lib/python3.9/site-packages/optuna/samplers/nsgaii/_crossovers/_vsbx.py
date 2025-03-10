from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class VSBXCrossover(BaseCrossover):
    """Modified Simulated Binary Crossover operation used by
    :class:`~optuna.samplers.NSGAIISampler`.

    vSBX generates child individuals without excluding any region of the parameter space,
    while maintaining the excellent properties of SBX.

    - `Pedro J. Ballester, Jonathan N. Carter.
      Real-Parameter Genetic Algorithms for Finding Multiple Optimal Solutions
      in Multi-modal Optimization. GECCO 2003: 706-717
      <https://doi.org/10.1007/3-540-45105-6_86>`__

    Args:
        eta:
            Distribution index. A small value of ``eta`` allows distant solutions
            to be selected as children solutions. If not specified, takes default
            value of ``2`` for single objective functions and ``20`` for multi objective.
    """

    n_parents = 2

    def __init__(self, eta: float | None = None) -> None:
        self._eta = eta

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:
        # https://doi.org/10.1007/3-540-45105-6_86
        # Section 3.2 Crossover Schemes (vSBX)
        if self._eta is None:
            eta = 20.0 if study._is_multi_objective() else 2.0
        else:
            eta = self._eta

        us = rng.rand(len(search_space_bounds))
        beta_1 = np.power(1 / 2 * us, 1 / (eta + 1))
        beta_2 = np.power(1 / 2 * (1 - us), 1 / (eta + 1))
        mask = us > 0.5
        c1 = 0.5 * ((1 + beta_1) * parents_params[0] + (1 - beta_1) * parents_params[1])
        c1[mask] = (
            0.5 * ((1 - beta_1) * parents_params[0] + (1 + beta_1) * parents_params[1])[mask]
        )
        c2 = 0.5 * ((3 - beta_2) * parents_params[0] - (1 - beta_2) * parents_params[1])
        c2[mask] = (
            0.5 * (-(1 - beta_2) * parents_params[0] + (3 - beta_2) * parents_params[1])[mask]
        )

        # vSBX applies crossover with establishment 0.5, and with probability 0.5,
        # the gene of the parent individual is the gene of the child individual.
        # The original SBX creates two child individuals,
        # but optuna's implementation creates only one child individual.
        # Therefore, when there is no crossover,
        # the gene is selected with equal probability from the parent individuals x1 and x2.

        child_params_list = []
        for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, parents_params[0], parents_params[1]):
            if rng.rand() < 0.5:
                if rng.rand() < 0.5:
                    child_params_list.append(c1_i)
                else:
                    child_params_list.append(c2_i)
            else:
                if rng.rand() < 0.5:
                    child_params_list.append(x1_i)
                else:
                    child_params_list.append(x2_i)
        child_params = np.array(child_params_list)

        return child_params
