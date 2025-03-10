from __future__ import annotations

import abc
from collections.abc import Sequence
import datetime
from typing import Any
from typing import overload

from optuna._deprecated import deprecated_func
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType


_SUGGEST_INT_POSITIONAL_ARGS = ["self", "name", "low", "high", "step", "log"]


class BaseTrial(abc.ABC):
    """Base class for trials.

    Note that this class is not supposed to be directly accessed by library users.
    """

    @abc.abstractmethod
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        raise NotImplementedError

    @deprecated_func("3.0.0", "6.0.0")
    @abc.abstractmethod
    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        raise NotImplementedError

    @deprecated_func("3.0.0", "6.0.0")
    @abc.abstractmethod
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        raise NotImplementedError

    @deprecated_func("3.0.0", "6.0.0")
    @abc.abstractmethod
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int:
        raise NotImplementedError

    @overload
    @abc.abstractmethod
    def suggest_categorical(self, name: str, choices: Sequence[None]) -> None: ...

    @overload
    @abc.abstractmethod
    def suggest_categorical(self, name: str, choices: Sequence[bool]) -> bool: ...

    @overload
    @abc.abstractmethod
    def suggest_categorical(self, name: str, choices: Sequence[int]) -> int: ...

    @overload
    @abc.abstractmethod
    def suggest_categorical(self, name: str, choices: Sequence[float]) -> float: ...

    @overload
    @abc.abstractmethod
    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str: ...

    @overload
    @abc.abstractmethod
    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType: ...

    @abc.abstractmethod
    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        raise NotImplementedError

    @abc.abstractmethod
    def report(self, value: float, step: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def should_prune(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def set_user_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    @deprecated_func("3.1.0", "5.0.0")
    def set_system_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def distributions(self) -> dict[str, BaseDistribution]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def user_attrs(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def system_attrs(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def datetime_start(self) -> datetime.datetime | None:
        raise NotImplementedError

    @property
    def number(self) -> int:
        raise NotImplementedError
