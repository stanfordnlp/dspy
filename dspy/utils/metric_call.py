"""Call metrics from DSPy's synchronous evaluation paths.

Metrics are ordinary callables. Sync metrics run on the calling thread; async
metrics (`async def` functions, objects with `async def __call__`, and sync
wrappers that return a coroutine) are awaited via :func:`asyncio.run` when no
event loop is running, and run on a daemon worker thread when a loop is already
running, so they cannot deadlock against the caller's loop.
"""

from __future__ import annotations

import asyncio
import inspect
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Callable

from dspy.dsp.utils.settings import thread_local_overrides


async def _await(result: Any) -> Any:
    return await result


def _run_in_worker_thread(func: Callable[[], Any]) -> Any:
    """Run ``func`` on a worker thread and return its result.

    Does not wait for a stuck worker after interruption, since the metric runs
    in the background otherwise.
    """
    context = copy_context()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(context.run, func)
    try:
        return future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def call_metric(metric: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a metric from synchronous code, accepting sync or async metrics.

    Sync metrics run on the calling thread. Async metrics (including objects
    with ``async def __call__`` and sync wrappers returning a coroutine) are
    awaited with :func:`asyncio.run` when no event loop is running on the
    calling thread; when a loop is already running, they run on a worker thread
    so their event-loop driving does not deadlock against the caller's loop.

    DSPy settings and context variables are copied into the worker thread, so
    async metrics still see the caller's configuration.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if not inspect.iscoroutinefunction(metric):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    asyncio.get_event_loop()
                except RuntimeError:
                    asyncio.set_event_loop(asyncio.new_event_loop())
            result = metric(*args, **kwargs)
            return result if not inspect.isawaitable(result) else asyncio.run(_await(result))
        result = metric(*args, **kwargs)
        return asyncio.run(_await(result)) if inspect.isawaitable(result) else result

    if inspect.iscoroutinefunction(metric):
        parent_overrides = thread_local_overrides.get().copy()

        def run_async() -> Any:
            original = thread_local_overrides.get()
            token = thread_local_overrides.set({**original, **parent_overrides})
            try:
                return asyncio.run(metric(*args, **kwargs))
            finally:
                thread_local_overrides.reset(token)

        return _run_in_worker_thread(run_async)

    result = metric(*args, **kwargs)
    if inspect.isawaitable(result):
        return _run_in_worker_thread(lambda: asyncio.run(_await(result)))
    return result
