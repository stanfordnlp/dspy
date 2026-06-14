import collections
import threading
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


def test_parallel_executor_tracks_failed_indices_and_exceptions():
    def task(item):
        if item == 3:
            raise ValueError("test error for 3")
        if item == 5:
            raise RuntimeError("test error for 5")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3, max_errors=3)

    results = executor.execute(task, data)

    assert results == [1, 2, None, 4, None]

    assert sorted(executor.failed_indices) == [2, 4]

    assert len(executor.exceptions_map) == 2
    assert isinstance(executor.exceptions_map[2], ValueError)
    assert str(executor.exceptions_map[2]) == "test error for 3"
    assert isinstance(executor.exceptions_map[4], RuntimeError)
    assert str(executor.exceptions_map[4]) == "test error for 5"


def test_sequential_execution_runs_on_main_thread():
    """With num_threads=1, all work should run on the main thread (not in a ThreadPoolExecutor)."""
    execution_threads = []

    def task(item):
        execution_threads.append(threading.current_thread())
        return item * 2

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=1)
    results = executor.execute(task, data)

    assert results == [2, 4, 6, 8, 10]
    assert all(t is threading.main_thread() for t in execution_threads)


def test_sequential_max_errors_not_met():
    """Sequential execution should handle errors without crashing when max_errors is not reached."""
    def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=1, max_errors=2)

    results = executor.execute(task, data)

    assert results == [1, 2, None, 4, 5]


def test_sequential_max_errors_exceeded():
    """Sequential execution should cancel when max_errors is reached."""
    def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=1, max_errors=1)

    with pytest.raises(Exception, match="Execution cancelled due to errors or interruption."):
        executor.execute(task, data)


def test_sequential_tracks_failed_indices_and_exceptions():
    """Sequential execution should track failed indices and exception objects."""
    def task(item):
        if item == 3:
            raise ValueError("test error for 3")
        if item == 5:
            raise RuntimeError("test error for 5")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=1, max_errors=3)

    results = executor.execute(task, data)

    assert results == [1, 2, None, 4, None]

    assert sorted(executor.failed_indices) == [2, 4]

    assert len(executor.exceptions_map) == 2
    assert isinstance(executor.exceptions_map[2], ValueError)
    assert str(executor.exceptions_map[2]) == "test error for 3"
    assert isinstance(executor.exceptions_map[4], RuntimeError)
    assert str(executor.exceptions_map[4]) == "test error for 5"


def test_straggler_resubmission_completes_without_error():
    """A task slower than the straggler timeout should still resolve (via resubmission)
    rather than crashing the executor."""
    def task(item):
        time.sleep(1.3 if item == 0 else 0.01)
        return item

    data = [0, 1, 2]
    executor = ParallelExecutor(num_threads=3, timeout=0.2, straggler_limit=3, disable_progress_bar=True)
    results = executor.execute(task, data)

    assert results == [0, 1, 2]


def test_straggler_resubmitted_at_most_once_per_item():
    """A persistently slow straggler must not be resubmitted unboundedly: the
    "at most once per item" invariant means the task is invoked at most twice."""
    calls = collections.Counter()
    lock = threading.Lock()

    def task(item):
        with lock:
            calls[item] += 1
        # item 0 stays slow long enough for its resubmission to also become a
        # straggler; without the fix it would be resubmitted again and again.
        time.sleep(2.6 if item == 0 else 0.01)
        return item

    data = [0, 1, 2]
    executor = ParallelExecutor(num_threads=4, timeout=0.2, straggler_limit=3, disable_progress_bar=True)
    results = executor.execute(task, data)

    assert results == [0, 1, 2]
    assert calls[0] <= 2, f"straggler resubmitted more than once (invoked {calls[0]} times)"


def test_timeout_zero_disables_straggler_resubmission():
    """With timeout=0, straggler resubmission is disabled: a slow task is invoked
    exactly once and the eval still completes."""
    calls = collections.Counter()
    lock = threading.Lock()

    def task(item):
        with lock:
            calls[item] += 1
        time.sleep(1.3 if item == 0 else 0.01)
        return item

    data = [0, 1, 2]
    executor = ParallelExecutor(num_threads=4, timeout=0, straggler_limit=3, disable_progress_bar=True)
    results = executor.execute(task, data)

    assert results == [0, 1, 2]
    assert calls[0] == 1


def test_sequential_compare_results():
    """Sequential execution should track and display comparison metrics correctly."""
    def task(item):
        return item, item > 2  # (result, score)

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=1, compare_results=True, disable_progress_bar=True)
    results = executor.execute(task, data)

    assert results == [(1, False), (2, False), (3, True), (4, True), (5, True)]
