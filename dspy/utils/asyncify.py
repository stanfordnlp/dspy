from typing import Any, Awaitable, Callable

import asyncer
from anyio import CapacityLimiter

from dspy.primitives.program import Module

_limiter = None


def get_async_max_workers():
    import dspy

    return dspy.settings.async_max_workers


def get_limiter():
    async_max_workers = get_async_max_workers()

    global _limiter
    if _limiter is None:
        _limiter = CapacityLimiter(async_max_workers)
    elif _limiter.total_tokens != async_max_workers:
        _limiter.total_tokens = async_max_workers

    return _limiter


def asyncify(program: Module) -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
    program in parallel with another task (e.g., another DSPy program).

    Args:
        program: The DSPy program to be wrapped for asynchronous execution.

    Returns:
        A function that takes the same arguments as the program, but returns an awaitable that
        resolves to the program's output.

    Example:
        >>> class TestSignature(dspy.Signature):
        >>>     input_text: str = dspy.InputField()
        >>>     output_text: str = dspy.OutputField()
        >>>
        >>> # Create the program and wrap it for asynchronous execution
        >>> program = dspy.asyncify(dspy.Predict(TestSignature))
        >>>
        >>> # Use the program asynchronously
        >>> async def get_prediction():
        >>>     prediction = await program(input_text="Test")
        >>>     print(prediction)  # Handle the result of the asynchronous execution
    """
    import threading

    assert threading.current_thread() is threading.main_thread(), "asyncify can only be called from the main thread"
    # NOTE: To allow this to be nested, we'd need behavior with contextvars like parallelizer.py
    return asyncer.asyncify(program, abandon_on_cancel=True, limiter=get_limiter())
