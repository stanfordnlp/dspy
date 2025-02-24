from typing import TYPE_CHECKING, Any, Awaitable, Callable

import asyncer
from anyio import CapacityLimiter

if TYPE_CHECKING:
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


def asyncify(program: "Module") -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
    program in parallel with another task (e.g., another DSPy program).

    This implementation propagates the current thread's configuration context to the worker thread.

    Args:
        program: The DSPy program to be wrapped for asynchronous execution.

    Returns:
        An async function: An async function that, when awaited, runs the program in a worker thread.
            The current thread's configuration context is inherited for each call.
    """

    async def async_program(*args, **kwargs) -> Any:
        # Capture the current overrides at call-time.
        from dspy.dsp.utils.settings import thread_local_overrides

        parent_overrides = thread_local_overrides.overrides.copy()

        def wrapped_program(*a, **kw):
            from dspy.dsp.utils.settings import thread_local_overrides

            original_overrides = thread_local_overrides.overrides
            thread_local_overrides.overrides = parent_overrides.copy()
            try:
                return program(*a, **kw)
            finally:
                thread_local_overrides.overrides = original_overrides

        # Create a fresh asyncified callable each time, ensuring the latest context is used.
        call_async = asyncer.asyncify(wrapped_program, abandon_on_cancel=True, limiter=get_limiter())
        return await call_async(*args, **kwargs)

    return async_program
