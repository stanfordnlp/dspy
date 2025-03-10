import enum
import inspect
import warnings
from functools import wraps
from typing import Callable, Optional

from .logging import get_logger


_emitted_deprecation_warnings = set()
logger = get_logger(__name__)


def deprecated(help_message: Optional[str] = None):
    """Decorator to mark a class or a function as deprecated.

    Args:
        help_message (:obj:`str`, optional): An optional message to guide the user on how to
            switch to non-deprecated usage of the library.
    """

    def decorator(deprecated_class_or_function: Callable):
        global _emitted_deprecation_warnings

        if inspect.isclass(deprecated_class_or_function):
            deprecated_function = deprecated_class_or_function.__init__
            name = deprecated_class_or_function.__name__
        else:
            deprecated_function = deprecated_class_or_function
            name = deprecated_function.__name__
            # Support deprecating __init__ class method: class name instead
            name = name if name != "__init__" else deprecated_function.__qualname__.split(".")[-2]

        warning_msg = (
            f"{name} is deprecated and will be removed in the next major version of datasets." + f" {help_message}"
            if help_message
            else ""
        )

        @wraps(deprecated_function)
        def wrapper(*args, **kwargs):
            func_hash = hash(deprecated_function)
            if func_hash not in _emitted_deprecation_warnings:
                warnings.warn(warning_msg, category=FutureWarning, stacklevel=2)
                _emitted_deprecation_warnings.add(func_hash)
            return deprecated_function(*args, **kwargs)

        wrapper._decorator_name_ = "deprecated"

        if inspect.isclass(deprecated_class_or_function):
            deprecated_class_or_function.__init__ = wrapper
            return deprecated_class_or_function
        else:
            return wrapper

    return decorator


class OnAccess(enum.EnumMeta):
    """
    Enum metaclass that calls a user-specified function whenever a member is accessed.
    """

    def __getattribute__(cls, name):
        obj = super().__getattribute__(name)
        if isinstance(obj, enum.Enum) and obj._on_access:
            obj._on_access()
        return obj

    def __getitem__(cls, name):
        member = super().__getitem__(name)
        if member._on_access:
            member._on_access()
        return member

    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        obj = super().__call__(value, names, module=module, qualname=qualname, type=type, start=start)
        if isinstance(obj, enum.Enum) and obj._on_access:
            obj._on_access()
        return obj


class DeprecatedEnum(enum.Enum, metaclass=OnAccess):
    """
    Enum class that calls `deprecate` method whenever a member is accessed.
    """

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        member._on_access = member.deprecate
        return member

    @property
    def help_message(self):
        return ""

    def deprecate(self):
        help_message = f" {self.help_message}" if self.help_message else ""
        warnings.warn(
            f"'{self.__objclass__.__name__}' is deprecated and will be removed in the next major version of datasets."
            + help_message,
            FutureWarning,
            stacklevel=3,
        )
