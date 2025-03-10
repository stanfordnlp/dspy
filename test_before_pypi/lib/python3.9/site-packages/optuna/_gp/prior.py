from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._gp import gp


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


DEFAULT_MINIMUM_NOISE_VAR = 1e-6


def default_log_prior(kernel_params: "gp.KernelParamsTensor") -> "torch.Tensor":
    # Log of prior distribution of kernel parameters.

    def gamma_log_prior(x: "torch.Tensor", concentration: float, rate: float) -> "torch.Tensor":
        # We omit the constant factor `rate ** concentration / Gamma(concentration)`.
        return (concentration - 1) * torch.log(x) - rate * x

    # NOTE(contramundum53): The priors below (params and function
    # shape for inverse_squared_lengthscales) were picked by heuristics.
    # TODO(contramundum53): Check whether these priors are appropriate.
    return (
        -(
            0.1 / kernel_params.inverse_squared_lengthscales
            + 0.1 * kernel_params.inverse_squared_lengthscales
        ).sum()
        + gamma_log_prior(kernel_params.kernel_scale, 2, 1)
        + gamma_log_prior(kernel_params.noise_var, 1.1, 30)
    )
