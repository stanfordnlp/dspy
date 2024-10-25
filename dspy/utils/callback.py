import functools
import inspect
import logging
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

import dspy

ACTIVE_CALL_ID = ContextVar("active_call_id", default=None)

logger = logging.getLogger(__name__)


class BaseCallback:
    """
    A base class for defining callback handlers for DSPy components.

    To use a callback, subclass this class and implement the desired handlers. Each handler
    will be called at the appropriate time before/after the execution of the corresponding component.

    For example, if you want to print a message before and after an LM is called, implement
    the on_llm_start and on_lm_end handler and set the callback to the global settings using `dspy.settings.configure`.

    ```
    import dspy
    from dspy.utils.callback import BaseCallback

    class LoggingCallback(BaseCallback):

        def on_lm_start(self, call_id, instance, inputs):
            print(f"LM is called with inputs: {inputs}")

        def on_lm_end(self, call_id, outputs, exception):
            print(f"LM is finished with outputs: {outputs}")

    dspy.settings.configure(
        callbacks=[LoggingCallback()]
    )

    cot = dspy.ChainOfThought("question -> answer")
    cot(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}
    ```

    Another way to set the callback is to pass it directly to the component constructor.
    In this case, the callback will only be triggered for that specific instance.

    ```
    lm = dspy.LM("gpt-3.5-turbo", callbacks=[LoggingCallback()])
    lm(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}

    lm_2 = dspy.LM("gpt-3.5-turbo")
    lm_2(question="What is the meaning of life?")
    # No logging here
    ```
    """

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        """
        A handler triggered when forward() method of a module (subclass of dspy.Module) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Module instance.
            inputs: The inputs to the module's forward() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_module_end(
        self,
        call_id: str,
        outputs: Optional[Any],
        exception: Optional[Exception] = None,
    ):
        """
        A handler triggered after forward() method of a module (subclass of dspy.Module) is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the module's forward() method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        """
        A handler triggered when __call__ method of dspy.LM instance is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The LM instance.
            inputs: The inputs to the LM's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_lm_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        """
        A handler triggered after __call__ method of dspy.LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the LM's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        """
        A handler triggered when format() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's format() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_format_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        """
        A handler triggered after format() method of dspy.LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's format() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        """
        A handler triggered when parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's parse() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_parse_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        """
        A handler triggered after parse() method of dspy.LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's parse() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass


def with_callbacks(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Combine global and local (per-instance) callbacks
        callbacks = dspy.settings.get("callbacks", []) + getattr(self, "callbacks", [])

        # if no callbacks are provided, just call the function
        if not callbacks:
            return fn(self, *args, **kwargs)

        # Generate call ID to connect start/end handlers if needed
        call_id = uuid.uuid4().hex

        inputs = inspect.getcallargs(fn, self, *args, **kwargs)
        inputs.pop("self")  # Not logging self as input

        for callback in callbacks:
            try:
                _get_on_start_handler(callback, self, fn)(call_id=call_id, instance=self, inputs=inputs)

            except Exception as e:
                logger.warning(f"Error when calling callback {callback}: {e}")

        results = None
        exception = None
        try:
            parent_call_id = ACTIVE_CALL_ID.get()
            # Active ID must be set right before the function is called,
            # not before calling the callbacks.
            ACTIVE_CALL_ID.set(call_id)
            results = fn(self, *args, **kwargs)
            return results
        except Exception as e:
            exception = e
            raise exception
        finally:
            ACTIVE_CALL_ID.set(parent_call_id)
            for callback in callbacks:
                try:
                    _get_on_end_handler(callback, self, fn)(
                        call_id=call_id,
                        outputs=results,
                        exception=exception,
                    )
                except Exception as e:
                    logger.warning(f"Error when calling callback {callback}: {e}")

    return wrapper


def _get_on_start_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """
    Selects the appropriate on_start handler of the callback
    based on the instance and function name.
    """
    if isinstance(instance, (dspy.LM)):
        return callback.on_lm_start
    elif isinstance(instance, (dspy.Adapter)):
        if fn.__name__ == "format":
            return callback.on_format_start
        elif fn.__name__ == "parse":
            return callback.on_parse_start

    # We treat everything else as a module.
    return callback.on_module_start


def _get_on_end_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """
    Selects the appropriate on_end handler of the callback
    based on the instance and function name.
    """
    if isinstance(instance, (dspy.LM)):
        return callback.on_lm_end
    elif isinstance(instance, (dspy.Adapter)):
        if fn.__name__ == "format":
            return callback.on_format_end
        elif fn.__name__ == "parse":
            return callback.on_parse_end

    # We treat everything else as a module.
    return callback.on_module_end
