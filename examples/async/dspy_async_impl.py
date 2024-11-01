import dspy
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio
import time

# Ideally:
os.environ["DSP_CACHEBOOL"] = "false"
lm = dspy.LM("gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm, async_mode=True)


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


async def async_main(num_items=200):
    module = dspy.ChainOfThought("question -> answer")

    start_time = time.time()
    tasks = [module(question=f"What is 1 + {i}?") for i in range(num_items)]
    await gather_with_concurrency(20, *tasks)
    end_time = time.time()

    print(f"Total time (async): {end_time - start_time} seconds")


def thread_main(num_items=200):
    module = dspy.ChainOfThought("question -> answer")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(
            executor.map(
                lambda x: module(question=x),
                [f"What is 1 + {i}?" for i in range(num_items)],
            )
        )
    end_time = time.time()

    print(f"Total time (threaded): {end_time - start_time} seconds")


if __name__ == "__main__":
    num_items = 500
    dspy.settings.configure(async_mode=False)
    thread_main(num_items=num_items)
    dspy.settings.configure(async_mode=True)
    # print("Sleeping for 60 seconds")
    # time.sleep(60)
    asyncio.run(async_main(num_items))

# edge cases I can think of:
# - multiple modules in the chain
# - async gets stuck/fails
# - what happens to settings inside each async module?
