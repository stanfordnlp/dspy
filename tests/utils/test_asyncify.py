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
