from time import time, sleep
import asyncio

import pytest

import dspy
from dspy.utils.asyncify import AsyncLimiter


@pytest.mark.anyio
async def test_async_limiter():
    standard_limiter = AsyncLimiter.get()
    assert standard_limiter.total_tokens == 8, "Default async capacity should be 8"
    assert AsyncLimiter.get() == standard_limiter, "AsyncLimiter should be a singleton"

    dspy.settings.configure(async_capacity=16)
    assert AsyncLimiter.get() != standard_limiter, "AsyncLimiter should be a new singleton"
    assert AsyncLimiter.get().total_tokens == 16, "Async capacity should be 16"
    assert AsyncLimiter.get() == AsyncLimiter.get(), "AsyncLimiter should be a singleton"


@pytest.mark.anyio
async def test_asyncify():
    def the_answer_to_life_the_universe_and_everything(wait: float):
        sleep(wait)
        return 42

    the_question = dspy.asyncify(the_answer_to_life_the_universe_and_everything)

    async def run_n_tasks(n: int, wait: float):
        await asyncio.gather(*[the_question(wait) for _ in range(n)])

    def set_async_capacity(capacity: int):
        dspy.settings.configure(async_capacity=capacity)

    async def assert_capacity(capacity: int, wait: float = 0.01, buffer: float = 0.1):
        set_async_capacity(capacity)
        t1 = time()
        await run_n_tasks(capacity, wait)
        t2 = time()
        assert t2 - t1 < wait + buffer

        t3 = time()
        await run_n_tasks(capacity * 1, wait)
        t4 = time()
        assert t4 - t3 < (wait + buffer) * 2

    await assert_capacity(1)
    await assert_capacity(2)
    await assert_capacity(4)
    await assert_capacity(8)
    await assert_capacity(16)
    await assert_capacity(32)
    await assert_capacity(64)
