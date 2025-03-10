from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import threading
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    import scipy.stats.qmc as qmc
else:
    from optuna._imports import _LazyImport

    qmc = _LazyImport("scipy.stats.qmc")


_threading_lock = threading.Lock()


class ScaleType(IntEnum):
    LINEAR = 0
    LOG = 1
    CATEGORICAL = 2


@dataclass(frozen=True)
class SearchSpace:
    scale_types: np.ndarray
    bounds: np.ndarray
    steps: np.ndarray


def unnormalize_one_param(
    param_value: np.ndarray, scale_type: ScaleType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    # param_value can be batched, or not.
    if scale_type == ScaleType.CATEGORICAL:
        return param_value
    low, high = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
    if scale_type == ScaleType.LOG:
        low, high = (math.log(low), math.log(high))
    param_value = param_value * (high - low) + low
    if scale_type == ScaleType.LOG:
        param_value = np.exp(param_value)
    return param_value


def normalize_one_param(
    param_value: np.ndarray, scale_type: ScaleType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    # param_value can be batched, or not.
    if scale_type == ScaleType.CATEGORICAL:
        return param_value
    low, high = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
    if scale_type == ScaleType.LOG:
        low, high = (math.log(low), math.log(high))
        param_value = np.log(param_value)
    if high == low:
        return np.full_like(param_value, 0.5)
    param_value = (param_value - low) / (high - low)
    return param_value


def round_one_normalized_param(
    param_value: np.ndarray, scale_type: ScaleType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    assert scale_type != ScaleType.CATEGORICAL
    if step == 0.0:
        return param_value

    param_value = unnormalize_one_param(param_value, scale_type, bounds, step)
    param_value = np.clip(
        (param_value - bounds[0] + 0.5 * step) // step * step + bounds[0],
        bounds[0],
        bounds[1],
    )
    param_value = normalize_one_param(param_value, scale_type, bounds, step)
    return param_value


def sample_normalized_params(
    n: int, search_space: SearchSpace, rng: np.random.RandomState | None
) -> np.ndarray:
    rng = rng or np.random.RandomState()
    dim = search_space.scale_types.shape[0]
    scale_types = search_space.scale_types
    bounds = search_space.bounds
    steps = search_space.steps

    # Sobol engine likely shares its internal state among threads.
    # Without threading.Lock, ValueError exceptions are raised in Sobol engine as discussed in
    # https://github.com/optuna/optunahub-registry/pull/168#pullrequestreview-2404054969
    with _threading_lock:
        qmc_engine = qmc.Sobol(dim, scramble=True, seed=rng.randint(np.iinfo(np.int32).max))
    param_values = qmc_engine.random(n)

    for i in range(dim):
        if scale_types[i] == ScaleType.CATEGORICAL:
            param_values[:, i] = np.floor(param_values[:, i] * bounds[i, 1])
        elif steps[i] != 0.0:
            param_values[:, i] = round_one_normalized_param(
                param_values[:, i], scale_types[i], (bounds[i, 0], bounds[i, 1]), steps[i]
            )
    return param_values


def get_search_space_and_normalized_params(
    trials: list[FrozenTrial],
    optuna_search_space: dict[str, BaseDistribution],
) -> tuple[SearchSpace, np.ndarray]:
    scale_types = np.zeros(len(optuna_search_space), dtype=np.int64)
    bounds = np.zeros((len(optuna_search_space), 2), dtype=np.float64)
    steps = np.zeros(len(optuna_search_space), dtype=np.float64)
    values = np.zeros((len(trials), len(optuna_search_space)), dtype=np.float64)
    for i, (param, distribution) in enumerate(optuna_search_space.items()):
        if isinstance(distribution, CategoricalDistribution):
            scale_types[i] = ScaleType.CATEGORICAL
            bounds[i, :] = (0.0, len(distribution.choices))
            steps[i] = 1.0
            values[:, i] = np.array(
                [distribution.to_internal_repr(trial.params[param]) for trial in trials]
            )
        else:
            assert isinstance(
                distribution,
                (
                    FloatDistribution,
                    IntDistribution,
                ),
            )
            scale_types[i] = ScaleType.LOG if distribution.log else ScaleType.LINEAR
            steps[i] = 0.0 if distribution.step is None else distribution.step
            bounds[i, :] = (distribution.low, distribution.high)

            values[:, i] = normalize_one_param(
                np.array([trial.params[param] for trial in trials]),
                scale_types[i],
                (bounds[i, 0], bounds[i, 1]),
                steps[i],
            )
    return SearchSpace(scale_types, bounds, steps), values


def get_unnormalized_param(
    optuna_search_space: dict[str, BaseDistribution],
    normalized_param: np.ndarray,
) -> dict[str, Any]:
    ret = {}
    for i, (param, distribution) in enumerate(optuna_search_space.items()):
        if isinstance(distribution, CategoricalDistribution):
            ret[param] = distribution.to_external_repr(normalized_param[i])
        else:
            assert isinstance(
                distribution,
                (
                    FloatDistribution,
                    IntDistribution,
                ),
            )
            scale_type = ScaleType.LOG if distribution.log else ScaleType.LINEAR
            step = 0.0 if distribution.step is None else distribution.step
            bounds = (distribution.low, distribution.high)
            param_value = float(
                np.clip(
                    unnormalize_one_param(normalized_param[i], scale_type, bounds, step),
                    distribution.low,
                    distribution.high,
                )
            )
            if isinstance(distribution, IntDistribution):
                param_value = round(param_value)
            ret[param] = param_value
    return ret
