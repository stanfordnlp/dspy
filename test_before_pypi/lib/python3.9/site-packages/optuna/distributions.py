from __future__ import annotations

import abc
from collections.abc import Sequence
import copy
import decimal
import json
from numbers import Real
from typing import Any
from typing import cast
from typing import Union
import warnings

import numpy as np

from optuna._deprecated import deprecated_class


CategoricalChoiceType = Union[None, bool, int, float, str]


_float_distribution_deprecated_msg = (
    "Use :class:`~optuna.distributions.FloatDistribution` instead."
)
_int_distribution_deprecated_msg = "Use :class:`~optuna.distributions.IntDistribution` instead."


class BaseDistribution(abc.ABC):
    """Base class for distributions.

    Note that distribution classes are not supposed to be called by library users.
    They are used by :class:`~optuna.trial.Trial` and :class:`~optuna.samplers` internally.
    """

    def to_external_repr(self, param_value_in_internal_repr: float) -> Any:
        """Convert internal representation of a parameter value into external representation.

        Args:
            param_value_in_internal_repr:
                Optuna's internal representation of a parameter value.

        Returns:
            Optuna's external representation of a parameter value.
        """

        return param_value_in_internal_repr

    @abc.abstractmethod
    def to_internal_repr(self, param_value_in_external_repr: Any) -> float:
        """Convert external representation of a parameter value into internal representation.

        Args:
            param_value_in_external_repr:
                Optuna's external representation of a parameter value.

        Returns:
            Optuna's internal representation of a parameter value.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def single(self) -> bool:
        """Test whether the range of this distribution contains just a single value.

        Returns:
            :obj:`True` if the range of this distribution contains just a single value,
            otherwise :obj:`False`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _contains(self, param_value_in_internal_repr: float) -> bool:
        """Test if a parameter value is contained in the range of this distribution.

        Args:
            param_value_in_internal_repr:
                Optuna's internal representation of a parameter value.

        Returns:
            :obj:`True` if the parameter value is contained in the range of this distribution,
            otherwise :obj:`False`.
        """

        raise NotImplementedError

    def _asdict(self) -> dict:
        return self.__dict__

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseDistribution):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((self.__class__,) + tuple(sorted(self.__dict__.items())))

    def __repr__(self) -> str:
        kwargs = ", ".join("{}={}".format(k, v) for k, v in sorted(self._asdict().items()))
        return "{}({})".format(self.__class__.__name__, kwargs)


class FloatDistribution(BaseDistribution):
    """A distribution on floats.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_float`, and passed to
    :mod:`~optuna.samplers` in general.

    .. note::
        When ``step`` is not :obj:`None`, if the range :math:`[\\mathsf{low}, \\mathsf{high}]`
        is not divisible by :math:`\\mathsf{step}`, :math:`\\mathsf{high}` will be replaced
        with the maximum of :math:`k \\times \\mathsf{step} + \\mathsf{low} < \\mathsf{high}`,
        where :math:`k` is an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
            ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,
            ``low`` must be larger than 0.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
            ``high`` must be greater than or equal to ``low``.
        log:
            If ``log`` is :obj:`True`, this distribution is in log-scaled domain.
            In this case, all parameters enqueued to the distribution must be positive values.
            This parameter must be :obj:`False` when the parameter ``step`` is not :obj:`None`.
        step:
            A discretization step. ``step`` must be larger than 0.
            This parameter must be :obj:`None` when the parameter ``log`` is :obj:`True`.

    """

    def __init__(
        self, low: float, high: float, log: bool = False, step: None | float = None
    ) -> None:
        if log and step is not None:
            raise ValueError("The parameter `step` is not supported when `log` is true.")

        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}).".format(low, high)
            )

        if log and low <= 0.0:
            raise ValueError(
                "The `low` value must be larger than 0 for a log distribution "
                "(low={}, high={}).".format(low, high)
            )

        if step is not None and step <= 0:
            raise ValueError(
                "The `step` value must be non-zero positive value, " "but step={}.".format(step)
            )

        self.step = None
        if step is not None:
            high = _adjust_discrete_uniform_high(low, high, step)
            self.step = float(step)

        self.low = float(low)
        self.high = float(high)
        self.log = log

    def single(self) -> bool:
        if self.step is None:
            return self.low == self.high
        else:
            if self.low == self.high:
                return True
            high = decimal.Decimal(str(self.high))
            low = decimal.Decimal(str(self.low))
            step = decimal.Decimal(str(self.step))
            return (high - low) < step

    def _contains(self, param_value_in_internal_repr: float) -> bool:
        value = param_value_in_internal_repr
        if self.step is None:
            return self.low <= value <= self.high
        else:
            k = (value - self.low) / self.step
            return self.low <= value <= self.high and abs(k - round(k)) < 1.0e-8

    def to_internal_repr(self, param_value_in_external_repr: float) -> float:
        try:
            internal_repr = float(param_value_in_external_repr)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"'{param_value_in_external_repr}' is not a valid type. "
                "float-castable value is expected."
            ) from e

        if np.isnan(internal_repr):
            raise ValueError(f"`{param_value_in_external_repr}` is invalid value.")
        if self.log and internal_repr <= 0.0:
            raise ValueError(
                f"`{param_value_in_external_repr}` is invalid value for the case log=True."
            )
        return internal_repr


@deprecated_class("3.0.0", "6.0.0", text=_float_distribution_deprecated_msg)
class UniformDistribution(FloatDistribution):
    """A uniform distribution in the linear domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_float`, and passed to
    :mod:`~optuna.samplers` in general.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
            ``low`` must be less than or equal to ``high``.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
            ``high`` must be greater than or equal to ``low``.

    """

    def __init__(self, low: float, high: float) -> None:
        super().__init__(low=low, high=high, log=False, step=None)

    def _asdict(self) -> dict:
        d = copy.deepcopy(self.__dict__)
        d.pop("log")
        d.pop("step")
        return d


@deprecated_class("3.0.0", "6.0.0", text=_float_distribution_deprecated_msg)
class LogUniformDistribution(FloatDistribution):
    """A uniform distribution in the log domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_float` with ``log=True``,
    and passed to :mod:`~optuna.samplers` in general.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
            ``low`` must be larger than 0. ``low`` must be less than or equal to ``high``.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
            ``high`` must be greater than or equal to ``low``.

    """

    def __init__(self, low: float, high: float) -> None:
        super().__init__(low=low, high=high, log=True, step=None)

    def _asdict(self) -> dict:
        d = copy.deepcopy(self.__dict__)
        d.pop("log")
        d.pop("step")
        return d


@deprecated_class("3.0.0", "6.0.0", text=_float_distribution_deprecated_msg)
class DiscreteUniformDistribution(FloatDistribution):
    """A discretized uniform distribution in the linear domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_float` with ``step``
    argument, and passed to :mod:`~optuna.samplers` in general.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by :math:`q`,
        :math:`\\mathsf{high}` will be replaced with the maximum of :math:`k q + \\mathsf{low}
        < \\mathsf{high}`, where :math:`k` is an integer.

    Args:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
            ``low`` must be less than or equal to ``high``.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
            ``high`` must be greater than or equal to ``low``.
        q:
            A discretization step. ``q`` must be larger than 0.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.

    """

    def __init__(self, low: float, high: float, q: float) -> None:
        super().__init__(low=low, high=high, step=q)

    def _asdict(self) -> dict:
        d = copy.deepcopy(self.__dict__)
        d.pop("log")

        step = d.pop("step")
        d["q"] = step
        return d

    @property
    def q(self) -> float:
        """Discretization step.

        :class:`~optuna.distributions.DiscreteUniformDistribution` is a subtype of
        :class:`~optuna.distributions.FloatDistribution`.
        This property is a proxy for its ``step`` attribute.
        """
        return cast(float, self.step)

    @q.setter
    def q(self, v: float) -> None:
        self.step = v


class IntDistribution(BaseDistribution):
    """A distribution on integers.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_int`, and passed to
    :mod:`~optuna.samplers` in general.

    .. note::
        When ``step`` is not :obj:`None`, if the range :math:`[\\mathsf{low}, \\mathsf{high}]`
        is not divisible by :math:`\\mathsf{step}`, :math:`\\mathsf{high}` will be replaced
        with the maximum of :math:`k \\times \\mathsf{step} + \\mathsf{low} < \\mathsf{high}`,
        where :math:`k` is an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
            ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,
            ``low`` must be larger than or equal to 1.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
            ``high`` must be greater than or equal to ``low``.
        log:
            If ``log`` is :obj:`True`, this distribution is in log-scaled domain.
            In this case, all parameters enqueued to the distribution must be positive values.
            This parameter must be :obj:`False` when the parameter ``step`` is not 1.
        step:
            A discretization step. ``step`` must be a positive integer. This parameter must be 1
            when the parameter ``log`` is :obj:`True`.

    """

    def __init__(self, low: int, high: int, log: bool = False, step: int = 1) -> None:
        if log and step != 1:
            raise ValueError(
                "Samplers and other components in Optuna only accept step is 1 "
                "when `log` argument is True."
            )

        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}).".format(low, high)
            )

        if log and low < 1:
            raise ValueError(
                "The `low` value must be equal to or greater than 1 for a log distribution "
                "(low={}, high={}).".format(low, high)
            )

        if step <= 0:
            raise ValueError(
                "The `step` value must be non-zero positive value, but step={}.".format(step)
            )

        self.log = log
        self.step = int(step)
        self.low = int(low)
        high = int(high)
        self.high = _adjust_int_uniform_high(self.low, high, self.step)

    def to_external_repr(self, param_value_in_internal_repr: float) -> int:
        return int(param_value_in_internal_repr)

    def to_internal_repr(self, param_value_in_external_repr: int) -> float:
        try:
            internal_repr = float(param_value_in_external_repr)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"'{param_value_in_external_repr}' is not a valid type. "
                "float-castable value is expected."
            ) from e

        if np.isnan(internal_repr):
            raise ValueError(f"`{param_value_in_external_repr}` is invalid value.")
        if self.log and internal_repr <= 0.0:
            raise ValueError(
                f"`{param_value_in_external_repr}` is invalid value for the case log=True."
            )
        return internal_repr

    def single(self) -> bool:
        if self.log:
            return self.low == self.high

        if self.low == self.high:
            return True
        return (self.high - self.low) < self.step

    def _contains(self, param_value_in_internal_repr: float) -> bool:
        value = param_value_in_internal_repr
        return self.low <= value <= self.high and (value - self.low) % self.step == 0


@deprecated_class("3.0.0", "6.0.0", text=_int_distribution_deprecated_msg)
class IntUniformDistribution(IntDistribution):
    """A uniform distribution on integers.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_int`, and passed to
    :mod:`~optuna.samplers` in general.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by
        :math:`\\mathsf{step}`, :math:`\\mathsf{high}` will be replaced with the maximum of
        :math:`k \\times \\mathsf{step} + \\mathsf{low} < \\mathsf{high}`, where :math:`k` is
        an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
            ``low`` must be less than or equal to ``high``.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
            ``high`` must be greater than or equal to ``low``.
        step:
            A discretization step. ``step`` must be a positive integer.

    """

    def __init__(self, low: int, high: int, step: int = 1) -> None:
        super().__init__(low=low, high=high, log=False, step=step)

    def _asdict(self) -> dict:
        d = copy.deepcopy(self.__dict__)
        d.pop("log")
        return d


@deprecated_class("3.0.0", "6.0.0", text=_int_distribution_deprecated_msg)
class IntLogUniformDistribution(IntDistribution):
    """A uniform distribution on integers in the log domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_int`, and passed to
    :mod:`~optuna.samplers` in general.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range
            and must be larger than or equal to 1. ``low`` must be less than or equal to ``high``.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
            ``high`` must be greater than or equal to ``low``.
        step:
            A discretization step. ``step`` must be a positive integer.

    """

    def __init__(self, low: int, high: int, step: int = 1) -> None:
        super().__init__(low=low, high=high, log=True, step=step)

    def _asdict(self) -> dict:
        d = copy.deepcopy(self.__dict__)
        d.pop("log")
        return d


def _categorical_choice_equal(
    value1: CategoricalChoiceType, value2: CategoricalChoiceType
) -> bool:
    """A function to check two choices equal considering NaN.

    This function can handle NaNs like np.float32("nan") other than float.
    """

    value1_is_nan = isinstance(value1, Real) and np.isnan(float(value1))
    value2_is_nan = isinstance(value2, Real) and np.isnan(float(value2))
    return (value1 == value2) or (value1_is_nan and value2_is_nan)


class CategoricalDistribution(BaseDistribution):
    """A categorical distribution.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_categorical`, and
    passed to :mod:`~optuna.samplers` in general.

    Args:
        choices:
            Parameter value candidates. ``choices`` must have one element at least.

    .. note::

        Not all types are guaranteed to be compatible with all storages. It is recommended to
        restrict the types of the choices to :obj:`None`, :class:`bool`, :class:`int`,
        :class:`float` and :class:`str`.

    Attributes:
        choices:
            Parameter value candidates.

    """

    def __init__(self, choices: Sequence[CategoricalChoiceType]) -> None:
        if len(choices) == 0:
            raise ValueError("The `choices` must contain one or more elements.")
        for choice in choices:
            if choice is not None and not isinstance(choice, (bool, int, float, str)):
                message = (
                    "Choices for a categorical distribution should be a tuple of None, bool, "
                    "int, float and str for persistent storage but contains {} which is of type "
                    "{}.".format(choice, type(choice).__name__)
                )
                warnings.warn(message)

        self.choices = tuple(choices)

    def to_external_repr(self, param_value_in_internal_repr: float) -> CategoricalChoiceType:
        return self.choices[int(param_value_in_internal_repr)]

    def to_internal_repr(self, param_value_in_external_repr: CategoricalChoiceType) -> float:
        try:
            # NOTE(nabenabe): With this implementation, we cannot distinguish some values
            # such as True and 1, or 1.0 and 1. For example, if choices=[True, 1] and external_repr
            # is 1, this method wrongly returns 0 instead of 1. However, we decided to accept this
            # bug for such exceptional choices for less complexity and faster processing.
            return self.choices.index(param_value_in_external_repr)
        except ValueError:  # ValueError: param_value_in_external_repr is not in choices.
            # ValueError also happens if external_repr is nan or includes precision error in float.
            for index, choice in enumerate(self.choices):
                if _categorical_choice_equal(param_value_in_external_repr, choice):
                    return index

        raise ValueError(f"'{param_value_in_external_repr}' not in {self.choices}.")

    def single(self) -> bool:
        return len(self.choices) == 1

    def _contains(self, param_value_in_internal_repr: float) -> bool:
        index = int(param_value_in_internal_repr)
        return 0 <= index < len(self.choices)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseDistribution):
            return NotImplemented
        if not isinstance(other, self.__class__):
            return False
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        for key, value in self.__dict__.items():
            if key == "choices":
                if len(value) != len(getattr(other, key)):
                    return False
                for choice, other_choice in zip(value, getattr(other, key)):
                    if not _categorical_choice_equal(choice, other_choice):
                        return False
            else:
                if value != getattr(other, key):
                    return False
        return True

    __hash__ = BaseDistribution.__hash__


DISTRIBUTION_CLASSES = (
    IntDistribution,
    IntLogUniformDistribution,
    IntUniformDistribution,
    FloatDistribution,
    UniformDistribution,
    LogUniformDistribution,
    DiscreteUniformDistribution,
    CategoricalDistribution,
)


def json_to_distribution(json_str: str) -> BaseDistribution:
    """Deserialize a distribution in JSON format.

    Args:
        json_str: A JSON-serialized distribution.

    Returns:
        A deserialized distribution.

    """

    json_dict = json.loads(json_str)

    if "name" in json_dict:
        if json_dict["name"] == CategoricalDistribution.__name__:
            json_dict["attributes"]["choices"] = tuple(json_dict["attributes"]["choices"])

        for cls in DISTRIBUTION_CLASSES:
            if json_dict["name"] == cls.__name__:
                return cls(**json_dict["attributes"])

        raise ValueError("Unknown distribution class: {}".format(json_dict["name"]))

    else:
        # Deserialize a distribution from an abbreviated format.
        if json_dict["type"] == "categorical":
            return CategoricalDistribution(json_dict["choices"])
        elif json_dict["type"] in ("float", "int"):
            low = json_dict["low"]
            high = json_dict["high"]
            step = json_dict.get("step")
            log = json_dict.get("log", False)

            if json_dict["type"] == "float":
                return FloatDistribution(low, high, log=log, step=step)

            else:
                if step is None:
                    step = 1
                return IntDistribution(low=low, high=high, log=log, step=step)

        raise ValueError("Unknown distribution type: {}".format(json_dict["type"]))


def distribution_to_json(dist: BaseDistribution) -> str:
    """Serialize a distribution to JSON format.

    Args:
        dist: A distribution to be serialized.

    Returns:
        A JSON string of a given distribution.

    """

    return json.dumps({"name": dist.__class__.__name__, "attributes": dist._asdict()})


def check_distribution_compatibility(
    dist_old: BaseDistribution, dist_new: BaseDistribution
) -> None:
    """A function to check compatibility of two distributions.

    It checks whether ``dist_old`` and ``dist_new`` are the same kind of distributions.
    If ``dist_old`` is :class:`~optuna.distributions.CategoricalDistribution`,
    it further checks ``choices`` are the same between ``dist_old`` and ``dist_new``.
    Note that this method is not supposed to be called by library users.

    Args:
        dist_old:
            A distribution previously recorded in storage.
        dist_new:
            A distribution newly added to storage.

    """

    if dist_old.__class__ != dist_new.__class__:
        raise ValueError("Cannot set different distribution kind to the same parameter name.")

    if isinstance(dist_old, (FloatDistribution, IntDistribution)):
        # For mypy.
        assert isinstance(dist_new, (FloatDistribution, IntDistribution))

        if dist_old.log != dist_new.log:
            raise ValueError("Cannot set different log configuration to the same parameter name.")

    if not isinstance(dist_old, CategoricalDistribution):
        return
    if not isinstance(dist_new, CategoricalDistribution):
        return
    if dist_old != dist_new:
        raise ValueError(
            CategoricalDistribution.__name__ + " does not support dynamic value space."
        )


def _adjust_discrete_uniform_high(low: float, high: float, step: float) -> float:
    d_high = decimal.Decimal(str(high))
    d_low = decimal.Decimal(str(low))
    d_step = decimal.Decimal(str(step))

    d_r = d_high - d_low

    if d_r % d_step != decimal.Decimal("0"):
        old_high = high
        high = float((d_r // d_step) * d_step + d_low)
        warnings.warn(
            "The distribution is specified by [{low}, {old_high}] and step={step}, but the range "
            "is not divisible by `step`. It will be replaced by [{low}, {high}].".format(
                low=low, old_high=old_high, high=high, step=step
            )
        )

    return high


def _adjust_int_uniform_high(low: int, high: int, step: int) -> int:
    r = high - low
    if r % step != 0:
        old_high = high
        high = r // step * step + low
        warnings.warn(
            "The distribution is specified by [{low}, {old_high}] and step={step}, but the range "
            "is not divisible by `step`. It will be replaced by [{low}, {high}].".format(
                low=low, old_high=old_high, high=high, step=step
            )
        )
    return high


def _get_single_value(distribution: BaseDistribution) -> int | float | CategoricalChoiceType:
    assert distribution.single()

    if isinstance(
        distribution,
        (
            FloatDistribution,
            IntDistribution,
        ),
    ):
        return distribution.low
    elif isinstance(distribution, CategoricalDistribution):
        return distribution.choices[0]
    assert False


# TODO(himkt): Remove this method with the deletion of deprecated distributions.
# https://github.com/optuna/optuna/issues/2941
def _convert_old_distribution_to_new_distribution(
    distribution: BaseDistribution,
    suppress_warning: bool = False,
) -> BaseDistribution:
    new_distribution: BaseDistribution

    # Float distributions.
    if isinstance(distribution, UniformDistribution):
        new_distribution = FloatDistribution(
            low=distribution.low,
            high=distribution.high,
            log=False,
            step=None,
        )
    elif isinstance(distribution, LogUniformDistribution):
        new_distribution = FloatDistribution(
            low=distribution.low,
            high=distribution.high,
            log=True,
            step=None,
        )
    elif isinstance(distribution, DiscreteUniformDistribution):
        new_distribution = FloatDistribution(
            low=distribution.low,
            high=distribution.high,
            log=False,
            step=distribution.q,
        )

    # Integer distributions.
    elif isinstance(distribution, IntUniformDistribution):
        new_distribution = IntDistribution(
            low=distribution.low,
            high=distribution.high,
            log=False,
            step=distribution.step,
        )
    elif isinstance(distribution, IntLogUniformDistribution):
        new_distribution = IntDistribution(
            low=distribution.low,
            high=distribution.high,
            log=True,
            step=distribution.step,
        )

    # Categorical distribution.
    else:
        new_distribution = distribution

    if new_distribution != distribution and not suppress_warning:
        message = (
            f"{distribution} is deprecated and internally converted to"
            f" {new_distribution}. See https://github.com/optuna/optuna/issues/2941."
        )
        warnings.warn(message, FutureWarning)

    return new_distribution


def _is_distribution_log(distribution: BaseDistribution) -> bool:
    if isinstance(distribution, (FloatDistribution, IntDistribution)):
        return distribution.log

    return False
