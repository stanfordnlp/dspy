from typing import Optional

from anyio import CapacityLimiter
import asyncer


class AsyncLimiter:
    _limiter: Optional[CapacityLimiter] = None

    @classmethod
    def get(cls):
        import dspy

        if cls._limiter is None or cls._limiter.total_tokens != dspy.settings.async_max_workers:
            cls._limiter = CapacityLimiter(dspy.settings.async_max_workers)

        return cls._limiter


def asyncify(program):
    return asyncer.asyncify(program, abandon_on_cancel=True, limiter=AsyncLimiter.get())
