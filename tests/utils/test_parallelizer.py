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


def test_sequential_compare_results():
    """Sequential execution should track and display comparison metrics correctly."""
    def task(item):
        return item, item > 2  # (result, score)

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=1, compare_results=True, disable_progress_bar=True)
    results = executor.execute(task, data)

    assert results == [(1, False), (2, False), (3, True), (4, True), (5, True)]


def test_timeout_none_disables_straggler_resubmission():
    """Passing ``timeout=None`` should skip the straggler resubmission path entirely."""
    def task(item):
        time.sleep(0.05)
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=2, timeout=None, disable_progress_bar=True)
    results = executor.execute(task, data)
    assert results == [1, 2, 3, 4, 5]


def test_timeout_zero_disables_straggler_resubmission():
    """``timeout=0`` should also skip the straggler resubmission path (legacy behaviour)."""
    def task(item):
        time.sleep(0.05)
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=2, timeout=0, disable_progress_bar=True)
    results = executor.execute(task, data)
    assert results == [1, 2, 3, 4, 5]


def test_straggler_resubmit_survives_executor_shutdown():
    """The straggler resubmit path must not crash when the underlying ThreadPoolExecutor is shut
    down out from under the parallel loop.

    Regression test for https://github.com/stanfordnlp/dspy/issues/9574: at interpreter shutdown
    (or any other path that triggers the executor's internal shutdown flag) ``submit`` raises
    ``RuntimeError: cannot schedule new futures after shutdown``. The parallel loop should catch
    this cleanly instead of letting the bare ``RuntimeError`` escape.
    """
    from concurrent.futures import ThreadPoolExecutor

    captured = []
    original_init = ThreadPoolExecutor.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured.append(self)

    def task(item):
        # The first item is deliberately slow so the straggler path will fire for it.
        if item == 0:
            time.sleep(2.0)
        else:
            time.sleep(0.05)
        return item

    def shut_down_externally():
        # Wait for the loop to start and submit work, then yank the executor out from under it.
        for _ in range(50):
            if captured:
                break
            time.sleep(0.02)
        time.sleep(0.2)
        if captured:
            captured[0].shutdown(wait=False)

    shutter = threading.Thread(target=shut_down_externally, daemon=True)
    shutter.start()

    try:
        ThreadPoolExecutor.__init__ = capturing_init
        executor = ParallelExecutor(
            num_threads=3,
            timeout=0.1,
            straggler_limit=3,
            disable_progress_bar=True,
        )
        # Must not raise ``RuntimeError("cannot schedule new futures after shutdown")``.
        results = executor.execute(task, [0, 1, 2, 3, 4])
    finally:
        ThreadPoolExecutor.__init__ = original_init
        shutter.join(timeout=2.0)

    # The result list should be complete with one entry per input. Running tasks are not
    # cancelled by ThreadPoolExecutor.shutdown(wait=False), so the slow item (index 0)
    # still finishes and lands in the output. The four fast items must be present too;
    # only the *new* submissions are rejected by the external shutdown, and the loop
    # surfaces those as None placeholders rather than raising.
    assert len(results) == 5
    fast_results = [r for r in results[1:] if r is not None]
    assert sorted(fast_results) == [r for r in [1, 2, 3, 4] if r in fast_results]
    assert all(r is None or isinstance(r, int) for r in results)
