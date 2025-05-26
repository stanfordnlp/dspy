import time

import pytest

from dspy.utils.parallelizer import ParallelExecutor


def test_worker_threads_independence():
    def task(item):
        # Each thread maintains its own state by appending to a thread-local list
        return item * 2

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3)
    results = executor.execute(task, data)

    assert results == [2, 4, 6, 8, 10]


def test_parallel_execution_speed():
    def task(item):
        time.sleep(0.1)  # Simulate a time-consuming task
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=5)

    start_time = time.time()
    executor.execute(task, data)
    end_time = time.time()

    assert end_time - start_time < len(data)


def test_max_errors_handling():
    def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3, max_errors=1)

    with pytest.raises(Exception, match="Execution cancelled due to errors or interruption."):
        executor.execute(task, data)


def test_max_errors_not_met():
    def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3, max_errors=2)

    # Ensure that the execution completes without crashing when max_errors is not met
    results = executor.execute(task, data)

    # Verify that the results exclude the failed task
    assert results == [1, 2, None, 4, 5]
