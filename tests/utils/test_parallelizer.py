import time

import pytest

import dspy
from dspy.utils.parallelizer import ParallelExecutor
from dspy.utils.usage_tracker import UsageTracker


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


def test_parallel_executor_with_usage_tracker_in_context():
    """Test that ParallelExecutor works correctly with UsageTracker inside dspy.context block.

    This tests the fix for the AttributeError: '_contextvars.ContextVar' object has no attribute 'overrides'
    bug that occurred when using UsageTracker with ParallelExecutor inside a dspy.context block.
    """
    tracker = UsageTracker()

    def task(item):
        return item * 2

    data = [1, 2, 3, 4, 5]

    # This should not raise AttributeError
    with dspy.context(usage_tracker=tracker):
        executor = ParallelExecutor(num_threads=2)
        results = executor.execute(task, data)

    assert results == [2, 4, 6, 8, 10]


def test_parallel_executor_with_usage_tracker_tracks_independently():
    """Test that each thread gets its own deep copy of the UsageTracker."""
    parent_tracker = UsageTracker()

    # Add some initial usage data to the parent tracker
    parent_tracker.add_usage("test-model", {"prompt_tokens": 100, "completion_tokens": 50})

    def task(item):
        from dspy.dsp.utils.settings import settings

        tracker = settings.usage_tracker
        # Each thread should have its own tracker (deep copy of parent)
        # Add usage specific to this thread
        tracker.add_usage("test-model", {"prompt_tokens": item, "completion_tokens": item})
        return (item, len(tracker.usage_data["test-model"]))

    data = [1, 2, 3]

    with dspy.context(usage_tracker=parent_tracker):
        executor = ParallelExecutor(num_threads=3)
        results = executor.execute(task, data)

    # Each thread should have its own tracker with 2 entries (1 from parent, 1 from thread)
    for item, entry_count in results:
        assert entry_count == 2, f"Thread processing item {item} should have 2 entries, got {entry_count}"

    # Parent tracker should only have its original entry (not modified by threads)
    assert len(parent_tracker.usage_data["test-model"]) == 1
