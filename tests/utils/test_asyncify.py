from time import time, sleep
import asyncio

import pytest

import dspy
from dspy.utils.asyncify import AsyncLimiter


@pytest.mark.anyio
async def test_async_limiter():
    limiter = AsyncLimiter.get()
    assert limiter.total_tokens == 8, "Default async capacity should be 8"
    assert AsyncLimiter.get() == limiter, "AsyncLimiter should be a singleton"

    dspy.settings.configure(async_max_workers=16)
    assert AsyncLimiter.get() == limiter, "AsyncLimiter should be a singleton"
    assert AsyncLimiter.get().total_tokens == 16, "Async capacity should be 16"
    assert AsyncLimiter.get() == AsyncLimiter.get(), "AsyncLimiter should be a singleton"


@pytest.mark.anyio
async def test_asyncify():
    def the_answer_to_life_the_universe_and_everything(wait: float):
        sleep(wait)
        return 42

    ask_the_question = dspy.asyncify(the_answer_to_life_the_universe_and_everything)

    async def run_n_tasks(n: int, wait: float):
        await asyncio.gather(*[ask_the_question(wait) for _ in range(n)])

    async def assert_capacity(capacity: int, wait: float = 0.01, buffer: float = 0.1):
        dspy.settings.configure(async_max_workers=capacity)
        t1 = time()
        await run_n_tasks(capacity, wait)
        t2 = time()
        assert t2 - t1 < wait + buffer

        t3 = time()
        await run_n_tasks(capacity * 1, wait)
        t4 = time()
        assert t4 - t3 < (wait + buffer) * 2

    for n in [1, 2, 4, 8, 16, 32, 64]:
        await assert_capacity(n)
