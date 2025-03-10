from __future__ import annotations

from optuna._experimental import experimental_func
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import FrozenTrial
from optuna.visualization.matplotlib import _matplotlib_imports


__all__ = ["is_available"]


@experimental_func("2.2.0")
def is_available() -> bool:
    """Returns whether visualization with Matplotlib is available or not.

    .. note::

        :mod:`~optuna.visualization.matplotlib` module depends on Matplotlib version 3.0.0 or
        higher. If a supported version of Matplotlib isn't installed in your environment, this
        function will return :obj:`False`. In such a case, please execute ``$ pip install -U
        matplotlib>=3.0.0`` to install Matplotlib.

    Returns:
        :obj:`True` if visualization with Matplotlib is available, :obj:`False` otherwise.
    """

    return _matplotlib_imports._imports.is_successful()


def _is_log_scale(trials: list[FrozenTrial], param: str) -> bool:
    for trial in trials:
        if param in trial.params:
            dist = trial.distributions[param]

            if isinstance(dist, (FloatDistribution, IntDistribution)):
                if dist.log:
                    return True

    return False


def _is_categorical(trials: list[FrozenTrial], param: str) -> bool:
    return any(
        isinstance(t.distributions[param], CategoricalDistribution)
        for t in trials
        if param in t.params
    )


def _is_numerical(trials: list[FrozenTrial], param: str) -> bool:
    return all(
        (isinstance(t.params[param], int) or isinstance(t.params[param], float))
        and not isinstance(t.params[param], bool)
        for t in trials
        if param in t.params
    )
