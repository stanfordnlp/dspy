import asyncio
import functools
import inspect
import threading
from typing import Any

import dspy.utils.callback_context as callback_context
from dspy.primitives import Example, Module
from dspy.utils.callback import with_callbacks


def _compile_invocation_key(instance):
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return id(instance), threading.get_ident(), id(task) if task is not None else None


def _with_compile_callbacks(fn):
    callback_fn = with_callbacks(fn)

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(instance, *args, **kwargs):
            active_compiles = callback_context._ACTIVE_COMPILES.get()
            invocation_key = _compile_invocation_key(instance)
            if invocation_key in active_compiles:
                return await fn(instance, *args, **kwargs)
            token = callback_context._ACTIVE_COMPILES.set((*active_compiles, invocation_key))
            try:
                return await callback_fn(instance, *args, **kwargs)
            finally:
                callback_context._ACTIVE_COMPILES.reset(token)

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(instance, *args, **kwargs):
        active_compiles = callback_context._ACTIVE_COMPILES.get()
        invocation_key = _compile_invocation_key(instance)
        if invocation_key in active_compiles:
            return fn(instance, *args, **kwargs)
        token = callback_context._ACTIVE_COMPILES.set((*active_compiles, invocation_key))
        try:
            return callback_fn(instance, *args, **kwargs)
        finally:
            callback_context._ACTIVE_COMPILES.reset(token)

    return wrapper


class Teleprompter:
    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Cover conventional class-body overrides while preserving inherited wrapped methods.
        # Replacing compile after class creation is not a supported instrumentation path.
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
