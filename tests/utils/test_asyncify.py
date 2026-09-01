import asyncio
import math
from time import sleep, time

import pytest

import dspy
from dspy.utils.asyncify import get_limiter


@pytest.mark.anyio
async def test_async_limiter():
    limiter = get_limiter()
    assert limiter.total_tokens == 8, "Default async capacity should be 8"
    assert get_limiter() == limiter, "AsyncLimiter should be a singleton"

    with dspy.context(async_max_workers=16):
        assert get_limiter() == limiter, "AsyncLimiter should be a singleton"
        assert get_limiter().total_tokens == 16, "Async capacity should be 16"
        assert get_limiter() == get_limiter(), "AsyncLimiter should be a singleton"


@pytest.mark.anyio
async def test_asyncify():
    def the_answer_to_life_the_universe_and_everything(wait: float):
        sleep(wait)
        return 42

    ask_the_question = dspy.asyncify(the_answer_to_life_the_universe_and_everything)

    async def run_n_tasks(n: int, wait: float):
        await asyncio.gather(*[ask_the_question(wait) for _ in range(n)])

    async def verify_asyncify(capacity: int, number_of_tasks: int, wait: float = 0.5):
        with dspy.context(async_max_workers=capacity):
            start = time()
            await run_n_tasks(number_of_tasks, wait)
            end = time()
            total_time = end - start

        # If asyncify is working correctly, the total time should be less than the total number of loops
        # `(number_of_tasks / capacity)` times wait time, plus the computational overhead. The lower bound should
        # be `math.floor(number_of_tasks * 1.0 / capacity) * wait` because there are more than
        # `math.floor(number_of_tasks * 1.0 / capacity)` loops.
        lower_bound = math.floor(number_of_tasks * 1.0 / capacity) * wait
        upper_bound = math.ceil(number_of_tasks * 1.0 / capacity) * wait + 2 * wait  # 2*wait for buffer

        assert lower_bound < total_time < upper_bound

    await verify_asyncify(4, 10)
    await verify_asyncify(8, 15)
    await verify_asyncify(8, 30)


def test_asyncify_return_type_is_variadic_callable():
    """Regression for #9058: `dspy.asyncify`'s return type annotation was
    `Callable[[Any, Any], Awaitable[Any]]`, claiming the wrapped function
    takes exactly two positional arguments. Pyright then raised
    `reportCallIssue: Expected 2 positional arguments` on any call shape
    other than exactly-two-args. Verify the annotation accepts variadic
    arguments via `Callable[..., Awaitable[Any]]`."""
    # `program: "Module"` is a forward ref behind `TYPE_CHECKING`, so
    # `typing.get_type_hints` can't resolve the function's annotations at
    # runtime. The bug is in the *return* annotation, which only references
    # public typing names — read it from `__annotations__` and evaluate
    # just that piece.
    import typing

    raw = dspy.asyncify.__annotations__["return"]
    if isinstance(raw, str):
        return_hint = eval(
            raw,
            {
                **vars(typing),
                "Any": typing.Any,
                "Awaitable": typing.Awaitable,
                "Callable": typing.Callable,
            },
        )
    else:
        return_hint = raw

    args = getattr(return_hint, "__args__", None)
    assert args is not None, f"return_hint has no __args__: {return_hint!r}"
    assert args[0] is Ellipsis, (
        f"asyncify return type should accept variadic args (...), got first arg: {args[0]!r}"
    )
