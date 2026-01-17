import time

import pytest
import threading

from dspy.utils.parallelizer import ParallelExecutor
from dspy.dsp.utils.settings import thread_local_overrides


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


def test_thread_local_overrides_with_usage_tracker():

    class MockUsageTracker:
        def __init__(self):
            self.tracked_items = []

        def track(self, value):
            self.tracked_items.append(value)

    parent_thread_usage_tracker = MockUsageTracker()
    parent_thread_overrides = {"usage_tracker": parent_thread_usage_tracker, "some_setting": "parent_value"}

    override_token = thread_local_overrides.set(parent_thread_overrides)

    try:
        worker_thread_ids = set()
        worker_thread_ids_lock = threading.Lock()

        # Track all usage tracker instances seen (may be same instance reused across tasks in same thread)
        all_usage_tracker_instances = []
        usage_tracker_instances_lock = threading.Lock()

        def task(item):

            current_thread_id = threading.get_ident()

            with worker_thread_ids_lock:
                worker_thread_ids.add(current_thread_id)

            current_thread_overrides = thread_local_overrides.get()

            # Verify overrides were copied to worker thread
            assert current_thread_overrides.get("some_setting") == "parent_value"

            worker_thread_usage_tracker = current_thread_overrides.get("usage_tracker")

            assert worker_thread_usage_tracker is not None
            assert isinstance(worker_thread_usage_tracker, MockUsageTracker)

            # Collect all tracker instances (same thread will get same instance)
            with usage_tracker_instances_lock:
                if worker_thread_usage_tracker not in all_usage_tracker_instances:
                    all_usage_tracker_instances.append(worker_thread_usage_tracker)

            worker_thread_usage_tracker.track(item)

            return item * 2

        input_data = [1, 2, 3, 4, 5]
        executor = ParallelExecutor(num_threads=3)
        results = executor.execute(task, input_data)

        assert results == [2, 4, 6, 8, 10]

        # Verify that worker threads got their own deep copied usage trackers
        # Even if only one thread was used, it should have a different instance than parent
        assert len(all_usage_tracker_instances) >= 1, "At least one worker usage tracker should exist"

        for worker_usage_tracker in all_usage_tracker_instances:
            assert worker_usage_tracker is not parent_thread_usage_tracker, (
                "Worker thread usage tracker should be deep copy, not same instance as parent"
            )

        assert len(parent_thread_usage_tracker.tracked_items) == 0, (
            "Parent usage tracker should not be modified by worker threads"
        )

        total_tracked_items_count = sum(len(tracker.tracked_items) for tracker in all_usage_tracker_instances)
        assert total_tracked_items_count == len(input_data), "All items should be tracked across worker threads"

    finally:
        thread_local_overrides.reset(override_token)
