import threading
import time
from unittest import mock

import pytest

from dspy.utils.callback import ACTIVE_CALL_ID
from dspy.utils.parallelizer import ParallelExecutor


def test_worker_threads_independence():
    def task(item):
        # Each thread maintains its own state by appending to a thread-local list
        return item * 2

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3)
    results = executor.execute(task, data)

    assert results == [2, 4, 6, 8, 10]


@pytest.mark.parametrize("num_threads", [1, 3])
def test_workers_inherit_active_callback_call_id(num_threads):
    executor = ParallelExecutor(num_threads=num_threads)

    for parent_call_id in ["first-parent", "second-parent"]:
        token = ACTIVE_CALL_ID.set(parent_call_id)
        try:
            observed_call_ids = executor.execute(lambda _: ACTIVE_CALL_ID.get(), range(5))

            assert observed_call_ids == [parent_call_id] * 5
            assert ACTIVE_CALL_ID.get() == parent_call_id
        finally:
            ACTIVE_CALL_ID.reset(token)


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


def test_sequential_compare_results():
    """Sequential execution should track and display comparison metrics correctly."""
    def task(item):
        return item, item > 2  # (result, score)

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=1, compare_results=True, disable_progress_bar=True)
    results = executor.execute(task, data)

    assert results == [(1, False), (2, False), (3, True), (4, True), (5, True)]


@pytest.mark.parametrize("num_threads", [1, 3])
def test_none_returning_tasks_are_counted_as_complete(num_threads):
    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=num_threads, disable_progress_bar=True)

    with mock.patch.object(executor, "_update_progress") as update_progress:
        results = executor.execute(lambda _: None, data)

    assert results == [None] * len(data)
    assert update_progress.call_args == mock.call(mock.ANY, len(data), len(data))


@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("fail_first", [False, True])
def test_tasks_isolate_inherited_usage_tracker(num_threads, fail_first):
    from dspy.dsp.utils.settings import settings
    from dspy.utils.usage_tracker import UsageTracker

    parent_tracker = UsageTracker()
    parent_tracker.add_usage("test-model", {"total_tokens": 5})

    def task(item):
        settings.usage_tracker.add_usage("test-model", {"total_tokens": 10})
        if fail_first and item == 0:
            raise ValueError("failed task")
        return settings.usage_tracker.get_total_tokens()

    executor = ParallelExecutor(num_threads=num_threads, max_errors=3, disable_progress_bar=True)
    with settings.context(usage_tracker=parent_tracker):
        results = executor.execute(task, [0, 1])
        assert settings.usage_tracker is parent_tracker

    expected = {"test-model": {"total_tokens": 15}}
    assert results == [None if fail_first else expected, expected]
    assert parent_tracker.get_total_tokens() == {"test-model": {"total_tokens": 5}}


def test_sequential_usage_tracker_restored_after_interrupt():
    from dspy.dsp.utils.settings import settings
    from dspy.utils.usage_tracker import UsageTracker

    parent_tracker = UsageTracker()

    def task(_):
        settings.usage_tracker.add_usage("test-model", {"total_tokens": 10})
        raise KeyboardInterrupt

    executor = ParallelExecutor(num_threads=1, disable_progress_bar=True)
    with settings.context(usage_tracker=parent_tracker):
        with pytest.raises(KeyboardInterrupt):
            executor.execute(task, [0])
        assert settings.usage_tracker is parent_tracker
        assert parent_tracker.get_total_tokens() == {}


@pytest.mark.parametrize("max_errors", [1, 3])
def test_sequential_tracker_copy_failure_respects_error_budget(max_errors):
    from dspy.dsp.utils.settings import settings, thread_local_overrides
    from dspy.utils.usage_tracker import UsageTracker

    parent_tracker = UsageTracker()
    task = mock.Mock(side_effect=lambda item: item)
    error = RuntimeError("tracker copy failed")
    executor = ParallelExecutor(num_threads=1, max_errors=max_errors, disable_progress_bar=True)

    with settings.context(usage_tracker=parent_tracker):
        original_overrides = thread_local_overrides.get()
        with mock.patch("dspy.utils.parallelizer.copy.deepcopy", side_effect=[error, UsageTracker()]):
            if max_errors == 1:
                with pytest.raises(Exception, match="Execution cancelled"):
                    executor.execute(task, [0, 1])
                task.assert_not_called()
            else:
                assert executor.execute(task, [0, 1]) == [None, 1]
                task.assert_called_once_with(1)
        assert thread_local_overrides.get() is original_overrides
        assert settings.usage_tracker is parent_tracker

    assert executor.failed_indices == [0]
    assert executor.exceptions_map == {0: error}
    assert executor.error_count == 1
    assert parent_tracker.get_total_tokens() == {}
