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

    assert threading.get_ident() == dspy.settings.main_tid, "asyncify can only be called from the main thread"

    def wrapped(*args, **kwargs):
        thread_stacks = dspy.settings.stack_by_thread
        current_thread_id = threading.get_ident()
        creating_new_thread = current_thread_id not in thread_stacks

        assert creating_new_thread
        thread_stacks[current_thread_id] = list(dspy.settings.main_stack)

        try:
            return program(*args, **kwargs)
        finally:
            del thread_stacks[threading.get_ident()]

    return asyncer.asyncify(wrapped, abandon_on_cancel=True, limiter=get_limiter())
