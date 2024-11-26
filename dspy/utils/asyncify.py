from anyio import CapacityLimiter
import asyncer


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


def asyncify(program):
    import dspy
    import threading
    assert threading.current_thread() is threading.main_thread() or \
                threading.current_thread().ident == dspy.settings.main_tid, \
                     "asyncify can only be called from the main thread"
                     
    # NOTE: To allow this to be nested, we'd need behavior with contextvars like parallelizer.py
    return asyncer.asyncify(program, abandon_on_cancel=True, limiter=get_limiter())
