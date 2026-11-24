"""Call metrics from DSPy's synchronous evaluation paths.

Metrics are ordinary callables, but they may be ``async def`` functions or may
drive their own event loop (e.g. an async LLM client wrapped in
``run_until_complete``). Running those directly on a thread that already has
an event loop — as happens in Jupyter — deadlocks. This module keeps the
synchronous metric API while making such metrics safe to use.
"""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Callable

from dspy.dsp.utils.settings import thread_local_overrides


async def _await(result: Any) -> Any:
    return await result


def call_metric(metric: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``metric`` from synchronous code, accepting sync or async metrics.

    When no event loop is running, sync metrics are called directly and async
    metrics are driven with :func:`asyncio.run`. When a loop is running, the
    metric runs on a worker thread with a fresh loop state, so metrics that
    create or drive their own loop cannot deadlock against the caller's loop.

    DSPy settings and context variables are copied into the worker thread, so
    metrics still see the caller's configuration and trace.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        result = metric(*args, **kwargs)
        return asyncio.run(_await(result)) if inspect.isawaitable(result) else result

    parent_overrides = thread_local_overrides.get().copy()
    context = copy_context()

    def worker() -> Any:
        original = thread_local_overrides.get()
        token = thread_local_overrides.set({**original, **parent_overrides})
        try:
            result = metric(*args, **kwargs)
            return asyncio.run(_await(result)) if inspect.isawaitable(result) else result
        finally:
            thread_local_overrides.reset(token)

    # A fresh thread has no running event loop, even when the caller (e.g. a
    # notebook) does. One worker is enough: the metric is the only task here.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(context.run, worker).result()
