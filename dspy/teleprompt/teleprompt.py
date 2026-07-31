import contextvars
import functools
from typing import Any

from dspy.primitives import Example, Module
from dspy.utils.callback import with_callbacks

_ACTIVE_OPTIMIZERS = contextvars.ContextVar("active_optimizers", default=())


def _with_compile_callbacks(fn):
    callback_fn = with_callbacks(fn)

    @functools.wraps(fn)
    def wrapper(instance, *args, **kwargs):
        if id(instance) in _ACTIVE_OPTIMIZERS.get():
            return fn(instance, *args, **kwargs)
        token = _ACTIVE_OPTIMIZERS.set((*_ACTIVE_OPTIMIZERS.get(), id(instance)))
        try:
            return callback_fn(instance, *args, **kwargs)
        finally:
            _ACTIVE_OPTIMIZERS.reset(token)

    return wrapper


class Teleprompter:
    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "compile" in cls.__dict__:
            cls.compile = _with_compile_callbacks(cls.compile)

    def compile(self, student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None, **kwargs) -> Module:
        """
        Optimize the student program.

        Args:
            student: The student program to optimize.
            trainset: The training set to use for optimization.
            teacher: The teacher program to use for optimization.
            valset: The validation set to use for optimization.

        Returns:
            The optimized student program.
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of the teleprompter.

        Returns:
            The parameters of the teleprompter.
        """
        return self.__dict__
