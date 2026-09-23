"""Cancellation-safe timeouts on every supported Python version.

Python 3.10/3.11 asyncio.wait_for can swallow caller cancellation when its
inner future completes concurrently (CPython #86296). Socket reads/writes
must not continue after that cancellation. Use asyncio.wait to distinguish
our timeout from caller cancellation, then explicitly cancel and drain the
owned operation. No global event-loop patches or third-party dependency.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

_T = TypeVar("_T")


async def wait_for(
    awaitable: Awaitable[_T], timeout: float | None,
    *, cancel_result: Callable[[_T], None] | None = None,
) -> _T:
    """Wait with a timeout, never replacing caller cancellation with success.

    Like asyncio.wait_for, timeout cancellation waits for the operation's
    cleanup. An operation that suppresses cancellation can return a value;
    an operation that suppresses it indefinitely can exceed the timeout.
    cancel_result releases a resource acquired concurrently with cancellation
    (a socket, or a semaphore permit) before its caller took ownership.
    """
    if timeout is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        if timeout > 0:
            done, _ = await asyncio.wait((task,), timeout=timeout)
            if done:
                return task.result()
        elif task.done():
            return task.result()
        task.cancel()
        # return_exceptions consumes only the child's cancellation. A caller
        # cancellation still raises here and reaches the outer handler.
        await asyncio.gather(task, return_exceptions=True)
        try:
            return task.result()
        except asyncio.CancelledError as exc:
            raise asyncio.TimeoutError() from exc
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if cancel_result is not None and not task.cancelled() and task.exception() is None:
            cancel_result(task.result())
        raise
