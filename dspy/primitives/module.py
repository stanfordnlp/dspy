import inspect
import logging
from typing import Any

from dspy.dsp.utils.settings import settings
from dspy.predict.parallel import Parallel
from dspy.primitives.base_module import BaseModule
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.utils import magicattr
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history
from dspy.utils.usage_tracker import track_usage

logger = logging.getLogger(__name__)


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
    """Base class for all DSPy modules (programs).

    A Module is a building block for DSPy programs that can contain predictors,
    sub-modules, and custom logic. Modules can be composed together to create
    complex pipelines and can be optimized using DSPy's teleprompters.

    All DSPy programs should inherit from this class and implement a ``forward``
    method that defines the program's logic.

    Args:
        callbacks: Optional list of callback handlers for instrumentation
            and monitoring.

    Attributes:
        callbacks: List of registered callback handlers.
        history: List of LM call history for this module.

    Example:
        >>> import dspy
        >>> class MyProgram(dspy.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.predictor = dspy.Predict("question -> answer")
        ...
        ...     def forward(self, question):
        ...         return self.predictor(question=question)
    """

    def _base_init(self):
        self._compiled = False
        self.callbacks = []
        self.history = []

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False
        # LM calling history of the module.
        self.history = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("history", None)
        state.pop("callbacks", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "history"):
            self.history = []
        if not hasattr(self, "callbacks"):
            self.callbacks = []

    @with_callbacks
    def __call__(self, *args, **kwargs) -> Prediction:
        from dspy.dsp.utils.settings import thread_local_overrides

        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = self.forward(*args, **kwargs)
                tokens = usage_tracker.get_total_tokens()
                self._set_lm_usage(tokens, output)

                return output

            return self.forward(*args, **kwargs)

    @with_callbacks
    async def acall(self, *args, **kwargs) -> Prediction:
        from dspy.dsp.utils.settings import thread_local_overrides

        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = await self.aforward(*args, **kwargs)
                    tokens = usage_tracker.get_total_tokens()
                    self._set_lm_usage(tokens, output)

                    return output

            return await self.aforward(*args, **kwargs)

    def named_predictors(self):
        """Return all named Predict modules in this module.

        Iterates through all parameters and returns those that are instances
        of ``dspy.Predict``, along with their names.

        Returns:
            list[tuple[str, Predict]]: A list of (name, predictor) tuples
                where name is the attribute path and predictor is the
                Predict instance.

        Example:
            >>> import dspy
            >>> class MyProgram(dspy.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.qa = dspy.Predict("question -> answer")
            ...         self.summarize = dspy.Predict("text -> summary")
            ...
            >>> program = MyProgram()
            >>> for name, p in program.named_predictors():
            ...     print(name)
            qa
            summarize
        """
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        """Return all Predict modules in this module.

        Returns:
            list[Predict]: A list of all Predict instances in this module.

        Example:
            >>> import dspy
            >>> class MyProgram(dspy.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.qa = dspy.Predict("question -> answer")
            ...
            >>> program = MyProgram()
            >>> len(program.predictors())
            1
        """
        return [param for _, param in self.named_predictors()]

    def set_lm(self, lm):
        """Set the language model for all predictors in this module.

        This method recursively sets the language model for all Predict
        instances contained within this module.

        Args:
            lm: The language model instance to use for all predictors.

        Example:
            >>> import dspy
            >>> lm = dspy.LM("openai/gpt-4o-mini")
            >>> program = dspy.Predict("question -> answer")
            >>> program.set_lm(lm)
        """
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        """Get the language model used by this module's predictors.

        Returns the language model if all predictors use the same LM.
        Raises an error if multiple different LMs are in use.

        Returns:
            The language model instance used by this module's predictors.

        Raises:
            ValueError: If multiple different language models are being
                used by the predictors in this module.
        """
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
        """Apply a function to all named predictors in this module.

        This method iterates through all Predict instances in the module
        and applies the given function to each, replacing the original
        predictor with the function's return value.

        Args:
            func: A callable that takes a Predict instance and returns
                a new Predict instance (or compatible object).

        Returns:
            Module: Returns self for method chaining.

        Example:
            >>> import dspy
            >>> class MyProgram(dspy.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.qa = dspy.Predict("question -> answer")
            ...
            >>> program = MyProgram()
            >>> program.map_named_predictors(lambda p: p)
        """
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    def inspect_history(self, n: int = 1):
        """Display the LM call history for this module.

        Prints a formatted view of the most recent language model calls
        made by this module, useful for debugging and understanding
        the module's behavior.

        Args:
            n: The number of recent history entries to display.
                Defaults to 1.

        Returns:
            The formatted history output.
        """
        return pretty_print_history(self.history, n)

    def batch(
        self,
        examples: list[Example],
        num_threads: int | None = None,
        max_errors: int | None = None,
        return_failed_examples: bool = False,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
    ) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]:
        """
        Processes a list of dspy.Example instances in parallel using the Parallel module.

        Args:
            examples: List of dspy.Example instances to process.
            num_threads: Number of threads to use for parallel processing.
            max_errors: Maximum number of errors allowed before stopping execution.
                If ``None``, inherits from ``dspy.settings.max_errors``.
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

    def _set_lm_usage(self, tokens: dict[str, Any], output: Any):
        # Some optimizers (e.g., GEPA bootstrap tracing) temporarily patch
        # module.forward to return a tuple: (prediction, trace).
        # When usage tracking is enabled, ensure we attach usage to the
        # prediction object if present.
        prediction_in_output = None
        if isinstance(output, Prediction):
            prediction_in_output = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Prediction):
            prediction_in_output = output[0]
        if prediction_in_output:
            prediction_in_output.set_lm_usage(tokens)
        else:
            logger.warning("Failed to set LM usage. Please return `dspy.Prediction` object from dspy.Module to enable usage tracking.")


    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name == "forward" and callable(attr):
            # Check if forward is called through __call__ or directly
            stack = inspect.stack()
            forward_called_directly = len(stack) <= 1 or stack[1].function != "__call__"

            if forward_called_directly:
                logger.warning(
                    f"Calling module.forward(...) on {self.__class__.__name__} directly is discouraged. "
                    f"Please use module(...) instead."
                )

        return attr


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)
