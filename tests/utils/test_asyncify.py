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
    def the_answer_to_life_the_universe_and_everything(started, release):
        with active_lock:
            active_workers[0] += 1
            max_active_workers[0] = max(max_active_workers[0], active_workers[0])
            if active_workers[0] == capacity:
                started.set()

        release.wait(timeout=2)

        with active_lock:
            active_workers[0] -= 1
        return 42

    ask_the_question = dspy.asyncify(the_answer_to_life_the_universe_and_everything)
    capacity = 4
    active_lock = threading.Lock()
    active_workers = [0]
    max_active_workers = [0]
    first_batch_started = threading.Event()
    release_workers = threading.Event()

    async def run_tasks():
        with dspy.context(async_max_workers=capacity):
            return await asyncio.gather(
                *[
                    ask_the_question(first_batch_started, release_workers)
                    for _ in range(capacity * 2)
                ]
            )

    tasks = asyncio.create_task(run_tasks())
    await asyncio.to_thread(first_batch_started.wait, 2)
    assert first_batch_started.is_set(), "The first asyncify worker batch should start"
    assert max_active_workers[0] == capacity

    release_workers.set()
    assert await asyncio.wait_for(tasks, timeout=2) == [42] * (capacity * 2)
    assert max_active_workers[0] == capacity
