from functools import wraps
import asyncio
import inspect

import dspy
from dspy.primitives.assertions import *
from dspy.primitives.module import BaseModule

import magicattr


class ProgramMeta(type):
    pass


def handle_async(func):
    """
    Decorator that handles both sync and async calls transparently.
    If the decorated function is called from an async context, runs async.
    If called from a sync context, runs sync.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # If we're not in an async context, run synchronously
        if not asyncio.iscoroutinefunction(func):
            return func(*args, **kwargs)

        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, return coroutine
            return func(*args, **kwargs)
        else:
            # We're not in an async context, run in new loop
            return asyncio.run(func(*args, **kwargs))

    return wrapper


class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False

    def __init__(self):
        self._compiled = False

    @handle_async
    def __call__(self, *args, **kwargs):
        if dspy.settings.async_mode:
            return self.forward_internal(*args, **kwargs)
        # print("Calling sync, ", *args, **kwargs)
        return self.forward(*args, **kwargs)

    async def forward_internal(self, *args, **kwargs):
        """Internal method that handles async execution of forward"""
        if hasattr(self, "forward"):
            # Convert all calls within forward to their async versions
            # with AsyncContext():
            result = self.forward(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        raise NotImplementedError("No forward method defined for this module")

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [
            (name, param)
            for name, param in self.named_parameters()
            if isinstance(param, Predict)
        ]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def set_lm(self, lm):
        if not dspy.settings.experimental:
            raise ValueError(
                "Setting or getting the LM of a program is an experimental feature. Please enable the "
                "'dspy.settings.experimental' flag to use these features."
            )

        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        if not dspy.settings.experimental:
            raise ValueError(
                "Setting or getting the LM of a program is an experimental feature. Please enable the "
                "'dspy.settings.experimental' flag to use these features."
            )

        all_used_lms = [param.lm for _, param in self.named_predictors()]

        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]

        raise ValueError("Multiple LMs are being used in the module.")

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def map_named_predictors(self, func):
        """Applies a function to all named predictors."""
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    def activate_assertions(self, handler=backtrack_handler, **handler_args):
        """
        Activates assertions for the module.
        The default handler is the backtrack_handler.
        """
        assert_transform_module(self, handler, **handler_args)
        return self

    # def __deepcopy__(self, memo):
    #     # memo is a dict of id's to copies already made during the current call
    #     # Check if the object is already copied
    #     if id(self) in memo:
    #         return memo[id(self)]

    #     print(f"Deep copying {self.__class__.__name__}...")

    #     new_copy = copy.copy(self)
    #     memo[id(self)] = new_copy

    #     for k, v in self.__dict__.items():
    #         print(f"Copying attribute {k} of type {type(v)}...")
    #         setattr(new_copy, k, copy.deepcopy(v, memo))
    #         print("Done")

    #     return new_copy


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)


Program = Module
