from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.logging import get_logger


if TYPE_CHECKING:
    import scipy.optimize as so
    import torch
else:
    from optuna._imports import _LazyImport

    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")

logger = get_logger(__name__)

# This GP implementation uses the following notation:
# X[len(trials), len(params)]: observed parameter values.
# Y[len(trials)]: observed objective values.
# x[(batch_len,) len(params)]: parameter value to evaluate. Possibly batched.
# cov_fX_fX[len(trials), len(trials)]: kernel matrix of X = V[f(X)]
# cov_fx_fX[(batch_len,) len(trials)]: kernel matrix of x and X = Cov[f(x), f(X)]
# cov_fx_fx: kernel value (scalar) of x = V[f(x)].
#     Since we use a Matern 5/2 kernel, we assume this value to be a constant.
# cov_Y_Y_inv[len(trials), len(trials)]: inv of the covariance matrix of Y = (V[f(X) + noise])^-1
# cov_Y_Y_inv_Y[len(trials)]: cov_Y_Y_inv @ Y
# max_Y: maximum of Y (Note that we transform the objective values such that it is maximized.)
# d2: squared distance between two points


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:  # type: ignore
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        # Notice that the derivative is taken w.r.t. d^2, but not w.r.t. d.
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function,
        # then deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        (deriv,) = ctx.saved_tensors
        return deriv * grad


def matern52_kernel_from_squared_distance(squared_distance: torch.Tensor) -> torch.Tensor:
    # sqrt5d = sqrt(5 * squared_distance)
    # exp(sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)
    #
    # We cannot let PyTorch differentiate the above expression because
    # the gradient runs into 0/0 at squared_distance=0.
    return Matern52Kernel.apply(squared_distance)  # type: ignore


@dataclass(frozen=True)
class KernelParamsTensor:
    # Kernel parameters to fit.
    inverse_squared_lengthscales: torch.Tensor  # [len(params)]
    kernel_scale: torch.Tensor  # Scalar
    noise_var: torch.Tensor  # Scalar


def kernel(
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParamsTensor,
    X1: torch.Tensor,  # [...batch_shape, n_A, len(params)]
    X2: torch.Tensor,  # [...batch_shape, n_B, len(params)]
) -> torch.Tensor:  # [...batch_shape, n_A, n_B]
    # kernel(x1, x2) = kernel_scale * matern52_kernel_from_squared_distance(
    #                     d2(x1, x2) * inverse_squared_lengthscales)
    # d2(x1, x2) = sum_i d2_i(x1_i, x2_i)
    # d2_i(x1_i, x2_i) = (x1_i - x2_i) ** 2  # if x_i is continuous
    # d2_i(x1_i, x2_i) = 1 if x1_i != x2_i else 0  # if x_i is categorical

    d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2

    # Use the Hamming distance for categorical parameters.
    d2[..., is_categorical] = (d2[..., is_categorical] > 0.0).type(torch.float64)
    d2 = (d2 * kernel_params.inverse_squared_lengthscales).sum(dim=-1)
    return matern52_kernel_from_squared_distance(d2) * kernel_params.kernel_scale


def kernel_at_zero_distance(
    kernel_params: KernelParamsTensor,
) -> torch.Tensor:  # [...batch_shape, n_A, n_B]
    # kernel(x, x) = kernel_scale
    return kernel_params.kernel_scale


def posterior(
    kernel_params: KernelParamsTensor,
    X: torch.Tensor,  # [len(trials), len(params)]
    is_categorical: torch.Tensor,  # bool[len(params)]
    cov_Y_Y_inv: torch.Tensor,  # [len(trials), len(trials)]
    cov_Y_Y_inv_Y: torch.Tensor,  # [len(trials)]
    x: torch.Tensor,  # [(batch,) len(params)]
) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: [(batch,)], var: [(batch,)])
    cov_fx_fX = kernel(is_categorical, kernel_params, x[..., None, :], X)[..., 0, :]
    cov_fx_fx = kernel_at_zero_distance(kernel_params)

    # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
    # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T
    mean = cov_fx_fX @ cov_Y_Y_inv_Y  # [batch]
    var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ cov_Y_Y_inv)).sum(dim=-1)  # [batch]
    # We need to clamp the variance to avoid negative values due to numerical errors.
    return (mean, torch.clamp(var, min=0.0))


def marginal_log_likelihood(
    X: torch.Tensor,  # [len(trials), len(params)]
    Y: torch.Tensor,  # [len(trials)]
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParamsTensor,
) -> torch.Tensor:  # Scalar
    # -0.5 * log((2pi)^n |C|) - 0.5 * Y^T C^-1 Y, where C^-1 = cov_Y_Y_inv
    # We apply the cholesky decomposition to efficiently compute log(|C|) and C^-1.

    cov_fX_fX = kernel(is_categorical, kernel_params, X, X)

    cov_Y_Y_chol = torch.linalg.cholesky(
        cov_fX_fX + kernel_params.noise_var * torch.eye(X.shape[0], dtype=torch.float64)
    )
    # log |L| = 0.5 * log|L^T L| = 0.5 * log|C|
    logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
    # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = Y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ Y
    cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[:, 0]
    return -0.5 * (
        logdet
        + X.shape[0] * math.log(2 * math.pi)
        # Y^T C^-1 Y = Y^T inv(L^T L) Y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
        + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
    )


def _fit_kernel_params(
    X: np.ndarray,  # [len(trials), len(params)]
    Y: np.ndarray,  # [len(trials)]
    is_categorical: np.ndarray,  # [len(params)]
    log_prior: Callable[[KernelParamsTensor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    initial_kernel_params: KernelParamsTensor,
    gtol: float,
) -> KernelParamsTensor:
    n_params = X.shape[1]

    # We apply log transform to enforce the positivity of the kernel parameters.
    # Note that we cannot just use the constraint because of the numerical unstability
    # of the marginal log likelihood.
    # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
    # pathological behavior of maximum likelihood estimation.
    initial_raw_params = np.concatenate(
        [
            np.log(initial_kernel_params.inverse_squared_lengthscales.detach().numpy()),
            [
                np.log(initial_kernel_params.kernel_scale.item()),
                # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
                np.log(initial_kernel_params.noise_var.item() - 0.99 * minimum_noise),
            ],
        ]
    )

    def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
        raw_params_tensor = torch.from_numpy(raw_params)
        raw_params_tensor.requires_grad_(True)
        with torch.enable_grad():
            params = KernelParamsTensor(
                inverse_squared_lengthscales=torch.exp(raw_params_tensor[:n_params]),
                kernel_scale=torch.exp(raw_params_tensor[n_params]),
                noise_var=(
                    torch.tensor(minimum_noise, dtype=torch.float64)
                    if deterministic_objective
                    else torch.exp(raw_params_tensor[n_params + 1]) + minimum_noise
                ),
            )
            loss = -marginal_log_likelihood(
                torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(is_categorical), params
            ) - log_prior(params)
            loss.backward()  # type: ignore
            # scipy.minimize requires all the gradients to be zero for termination.
            raw_noise_var_grad = raw_params_tensor.grad[n_params + 1]  # type: ignore
            assert not deterministic_objective or raw_noise_var_grad == 0
        return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

    # jac=True means loss_func returns the gradient for gradient descent.
    res = so.minimize(
        # Too small `gtol` causes instability in loss_func optimization.
        loss_func,
        initial_raw_params,
        jac=True,
        method="l-bfgs-b",
        options={"gtol": gtol},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    raw_params_opt_tensor = torch.from_numpy(res.x)

    res = KernelParamsTensor(
        inverse_squared_lengthscales=torch.exp(raw_params_opt_tensor[:n_params]),
        kernel_scale=torch.exp(raw_params_opt_tensor[n_params]),
        noise_var=(
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else minimum_noise + torch.exp(raw_params_opt_tensor[n_params + 1])
        ),
    )
    return res


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[KernelParamsTensor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    initial_kernel_params: KernelParamsTensor | None = None,
    gtol: float = 1e-2,
) -> KernelParamsTensor:
    default_initial_kernel_params = KernelParamsTensor(
        inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
        kernel_scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(1.0, dtype=torch.float64),
    )
    if initial_kernel_params is None:
        initial_kernel_params = default_initial_kernel_params

    error = None
    # First try optimizing the kernel params with the provided initial_kernel_params,
    # but if it fails, rerun the optimization with the default initial_kernel_params.
    # This increases the robustness of the optimization.
    for init_kernel_params in [initial_kernel_params, default_initial_kernel_params]:
        try:
            return _fit_kernel_params(
                X=X,
                Y=Y,
                is_categorical=is_categorical,
                log_prior=log_prior,
                minimum_noise=minimum_noise,
                initial_kernel_params=init_kernel_params,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as e:
            error = e

    logger.warning(
        f"The optimization of kernel_params failed: \n{error}\n"
        "The default initial kernel params will be used instead."
    )
    return default_initial_kernel_params
