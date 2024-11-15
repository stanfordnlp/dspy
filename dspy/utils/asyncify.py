import sys
from typing import Awaitable, Optional, TypeVar, Callable

from anyio import CapacityLimiter
import asyncer

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


class AsyncLimiter:
    _limiter: Optional[CapacityLimiter] = None

    @classmethod
    def get(cls):
        import dspy

        if cls._limiter is None or cls._limiter.total_tokens != dspy.settings.max_async_workers:
            cls._limiter = CapacityLimiter(dspy.settings.max_async_workers)

        return cls._limiter


def asyncify(
    program: Callable[T_ParamSpec, T_Retval],
) -> Callable[T_ParamSpec, Awaitable[T_Retval]]:
    return asyncer.asyncify(program, abandon_on_cancel=True, limiter=AsyncLimiter.get())
