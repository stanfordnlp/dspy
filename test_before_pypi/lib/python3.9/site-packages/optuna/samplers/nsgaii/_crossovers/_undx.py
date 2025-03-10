from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class UNDXCrossover(BaseCrossover):
    """Unimodal Normal Distribution Crossover used by :class:`~optuna.samplers.NSGAIISampler`.

    Generates child individuals from the three parents
    using a multivariate normal distribution.

    - `H. Kita, I. Ono and S. Kobayashi,
      Multi-parental extension of the unimodal normal distribution crossover
      for real-coded genetic algorithms,
      Proceedings of the 1999 Congress on Evolutionary Computation-CEC99
      (Cat. No. 99TH8406), 1999, pp. 1581-1588 Vol. 2
      <https://doi.org/10.1109/CEC.1999.782672>`__

    Args:
        sigma_xi:
            Parametrizes normal distribution from which ``xi`` is drawn.
        sigma_eta:
            Parametrizes normal distribution from which ``etas`` are drawn.
            If not specified, defaults to ``0.35 / sqrt(len(search_space))``.
    """

    n_parents = 3

    def __init__(self, sigma_xi: float = 0.5, sigma_eta: float | None = None) -> None:
        self._sigma_xi = sigma_xi
        self._sigma_eta = sigma_eta

    def _distance_from_x_to_psl(self, parents_params: np.ndarray) -> np.floating:
        # The line connecting x1 to x2 is called psl (primary search line).
        # Compute the 2-norm of the vector orthogonal to psl from x3.
        e_12 = UNDXCrossover._normalized_x1_to_x2(
            parents_params
        )  # Normalized vector from x1 to x2.
        v_13 = parents_params[2] - parents_params[0]  # Vector from x1 to x3.
        v_12_3 = v_13 - np.dot(v_13, e_12) * e_12  # Vector orthogonal to v_12 through x3.
        m_12_3 = np.linalg.norm(v_12_3, ord=2)  # 2-norm of v_12_3.

        return m_12_3

    def _orthonormal_basis_vector_to_psl(self, parents_params: np.ndarray, n: int) -> np.ndarray:
        # Compute orthogonal basis vectors for the subspace orthogonal to psl.
        e_12 = UNDXCrossover._normalized_x1_to_x2(
            parents_params
        )  # Normalized vector from x1 to x2.
        basis_matrix = np.identity(n)

        if np.count_nonzero(e_12) != 0:
            basis_matrix[0] = e_12

        basis_matrix_t = basis_matrix.T
        Q, _ = np.linalg.qr(basis_matrix_t)

        return Q.T[1:]

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:
        # https://doi.org/10.1109/CEC.1999.782672
        # Section 2 Unimodal Normal Distribution Crossover
        n = len(search_space_bounds)
        xp = (parents_params[0] + parents_params[1]) / 2  # Section 2 (2).
        d = parents_params[0] - parents_params[1]  # Section 2 (3).
        if self._sigma_eta is None:
            sigma_eta = 0.35 / np.sqrt(n)
        else:
            sigma_eta = self._sigma_eta

        etas = rng.normal(0, sigma_eta**2, size=n)
        xi = rng.normal(0, self._sigma_xi**2)
        es = self._orthonormal_basis_vector_to_psl(
            parents_params, n
        )  # Orthonormal basis vectors of the subspace orthogonal to the psl.
        one = xp  # Section 2 (5).
        two = xi * d  # Section 2 (5).

        if n > 1:  # When n=1, there is no subsearch component.
            three = np.zeros(n)  # Section 2 (5).
            D = self._distance_from_x_to_psl(parents_params)  # Section 2 (4).
            for i in range(n - 1):
                three += etas[i] * es[i]
            three *= D
            child_params = one + two + three

        else:
            child_params = one + two

        return child_params

    @staticmethod
    def _normalized_x1_to_x2(parents_params: np.ndarray) -> np.ndarray:
        # Compute the normalized vector from x1 to x2.
        v_12 = parents_params[1] - parents_params[0]
        m_12 = np.linalg.norm(v_12, ord=2)
        e_12 = v_12 / np.clip(m_12, 1e-10, None)

        return e_12
