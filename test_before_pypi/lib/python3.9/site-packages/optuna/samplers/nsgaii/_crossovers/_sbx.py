from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class SBXCrossover(BaseCrossover):
    """Simulated Binary Crossover operation used by :class:`~optuna.samplers.NSGAIISampler`.

    Generates a child from two parent individuals
    according to the polynomial probability distribution.

    - `Deb, K. and R. Agrawal.
      “Simulated Binary Crossover for Continuous Search Space.”
      Complex Syst. 9 (1995): n. pag.
      <https://www.complex-systems.com/abstracts/v09_i02_a02/>`__

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
        # https://www.researchgate.net/profile/M-M-Raghuwanshi/publication/267198495_Simulated_Binary_Crossover_with_Lognormal_Distribution/links/5576c78408ae7536375205d7/Simulated-Binary-Crossover-with-Lognormal-Distribution.pdf
        # Section 2 Simulated Binary Crossover (SBX)

        # To avoid generating solutions that violate the box constraints,
        # alpha1, alpha2, xls and xus are introduced, unlike the reference.
        xls = search_space_bounds[..., 0]
        xus = search_space_bounds[..., 1]

        xs_min = np.min(parents_params, axis=0)
        xs_max = np.max(parents_params, axis=0)
        if self._eta is None:
            eta = 20.0 if study._is_multi_objective() else 2.0
        else:
            eta = self._eta

        xs_diff = np.clip(xs_max - xs_min, 1e-10, None)
        beta1 = 1 + 2 * (xs_min - xls) / xs_diff
        beta2 = 1 + 2 * (xus - xs_max) / xs_diff
        alpha1 = 2 - np.power(beta1, -(eta + 1))
        alpha2 = 2 - np.power(beta2, -(eta + 1))

        us = rng.rand(len(search_space_bounds))
        mask1 = us > 1 / alpha1  # Equation (3).
        betaq1 = np.power(us * alpha1, 1 / (eta + 1))  # Equation (3).
        betaq1[mask1] = np.power((1 / (2 - us * alpha1)), 1 / (eta + 1))[mask1]  # Equation (3).

        mask2 = us > 1 / alpha2  # Equation (3).
        betaq2 = np.power(us * alpha2, 1 / (eta + 1))  # Equation (3)
        betaq2[mask2] = np.power((1 / (2 - us * alpha2)), 1 / (eta + 1))[mask2]  # Equation (3).

        c1 = 0.5 * ((xs_min + xs_max) - betaq1 * xs_diff)  # Equation (4).
        c2 = 0.5 * ((xs_min + xs_max) + betaq2 * xs_diff)  # Equation (5).

        # SBX applies crossover with establishment 0.5, and with probability 0.5,
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
