import asyncio
import threading

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
    release = threading.Event()
    workers_started = threading.Event()
    worker_count_lock = threading.Lock()
    started_workers = 0
    expected_capacity = 0

    def the_answer_to_life_the_universe_and_everything():
        nonlocal started_workers
        with worker_count_lock:
            started_workers += 1
            if started_workers == expected_capacity:
                workers_started.set()
        release.wait()
        return 42

    ask_the_question = dspy.asyncify(the_answer_to_life_the_universe_and_everything)

    async def verify_asyncify(capacity: int, number_of_tasks: int):
        nonlocal expected_capacity, started_workers
        expected_capacity = capacity
        started_workers = 0
        release.clear()
        workers_started.clear()
        with dspy.context(async_max_workers=capacity):
            limiter = get_limiter()
            tasks = [asyncio.create_task(ask_the_question()) for _ in range(number_of_tasks)]

            async def wait_until_full():
                while limiter.borrowed_tokens < capacity:
                    await asyncio.sleep(0)

            try:
                await asyncio.wait_for(wait_until_full(), timeout=2)
                assert await asyncio.to_thread(workers_started.wait, 2)
                assert started_workers == capacity
                assert limiter.borrowed_tokens == capacity
            finally:
                release.set()

            assert await asyncio.gather(*tasks) == [42] * number_of_tasks

    await verify_asyncify(4, 10)
    await verify_asyncify(8, 15)
    await verify_asyncify(8, 30)
