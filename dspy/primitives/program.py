from typing import Optional

import magicattr

from dspy.dsp.utils.settings import settings
from dspy.predict.parallel import Parallel
from dspy.primitives.module import BaseModule
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history
from dspy.utils.usage_tracker import track_usage


class ProgramMeta(type):
    """Metaclass ensuring every ``dspy.Module`` instance is properly initialised."""

    def __call__(cls, *args, **kwargs):
        # Create the instance without invoking ``__init__`` so we can inject
        # the base initialization beforehand.
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            # ``_base_init`` sets attributes that should exist on all modules
            # even when a subclass forgets to call ``super().__init__``.
            Module._base_init(obj)
            cls.__init__(obj, *args, **kwargs)

            # Guarantee existence of critical attributes if ``__init__`` didn't
            # create them.
            if not hasattr(obj, "callbacks"):
                obj.callbacks = []
            if not hasattr(obj, "history"):
                obj.history = []
        return obj


class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False
        self.callbacks = []
        self.history = []

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False
        # LM calling history of the module.
        self.history = []

    @with_callbacks
    def __call__(self, *args, **kwargs):
        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and settings.usage_tracker is None:
                with track_usage() as usage_tracker:
                    output = self.forward(*args, **kwargs)
                output.set_lm_usage(usage_tracker.get_total_tokens())
                return output

            return self.forward(*args, **kwargs)

    @with_callbacks
    async def acall(self, *args, **kwargs):
        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and settings.usage_tracker is None:
                with track_usage() as usage_tracker:
                    output = await self.aforward(*args, **kwargs)
                    output.set_lm_usage(usage_tracker.get_total_tokens())
                    return output

            return await self.aforward(*args, **kwargs)

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def set_lm(self, lm):
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        all_used_lms = [param.lm for _, param in self.named_predictors()]

        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]

        raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

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

    def inspect_history(self, n: int = 1):
        return pretty_print_history(self.history, n)

    def batch(
        self,
        examples,
        num_threads: Optional[int] = None,
        max_errors: int = 10,
        return_failed_examples: bool = False,
        provide_traceback: Optional[bool] = None,
        disable_progress_bar: bool = False,
    ):
        """
        Processes a list of dspy.Example instances in parallel using the Parallel module.

        Args:
            examples: List of dspy.Example instances to process.
            num_threads: Number of threads to use for parallel processing.
            max_errors: Maximum number of errors allowed before stopping execution.
            return_failed_examples: Whether to return failed examples and exceptions.
            provide_traceback: Whether to include traceback information in error logs.
            disable_progress_bar: Whether to display the progress bar.
            
        Returns:
            List of results, and optionally failed examples and exceptions.
        """
        # Create a list of execution pairs (self, example)
        exec_pairs = [(self, example.inputs()) for example in examples]

        # Create an instance of Parallel
        parallel_executor = Parallel(
            num_threads=num_threads,
            max_errors=max_errors,
            return_failed_examples=return_failed_examples,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )

        # Execute the forward method of Parallel
        if return_failed_examples:
            results, failed_examples, exceptions = parallel_executor.forward(exec_pairs)
            return results, failed_examples, exceptions
        else:
            results = parallel_executor.forward(exec_pairs)
            return results


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)


Program = Module
