import threading
import time
from unittest import mock

import pytest

import dspy
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


def test_parallel_executor_records_harness_level_exceptions(monkeypatch):
    """An exception raised by the executor's own per-worker setup (e.g. deep-copying a
    thread-local override), as opposed to the user function, must still be tracked and
    logged like any other failure instead of silently leaving the result as None."""
    import dspy.utils.parallelizer as parallelizer_module

    def broken_deepcopy(*_args, **_kwargs):
        raise RuntimeError("cannot deepcopy usage_tracker")

    monkeypatch.setattr(parallelizer_module.copy, "deepcopy", broken_deepcopy)

    def task(item):
        return item

    data = [1, 2, 3]
    executor = ParallelExecutor(num_threads=3, max_errors=10)

    with dspy.context(usage_tracker=object()):
        results = executor.execute(task, data)

    assert results == [None, None, None]
    assert sorted(executor.failed_indices) == [0, 1, 2]
    assert len(executor.exceptions_map) == 3
    assert all(isinstance(e, RuntimeError) for e in executor.exceptions_map.values())


def test_straggler_retry_does_not_double_count_harness_exception(monkeypatch):
    """A straggler's original future and its resubmitted retry can both fail during harness
    setup for the SAME logical input. results[idx] stays None for an exception outcome (see
    _process_outcome), so that check alone can't distinguish "not yet recorded" from "already
    recorded by the other future" -- the retry's failure must not be counted a second time."""
    import dspy.utils.parallelizer as parallelizer_module

    call_count = {"n": 0}
    call_lock = threading.Lock()

    def broken_deepcopy(*_args, **_kwargs):
        with call_lock:
            call_count["n"] += 1
            is_original = call_count["n"] == 1
        if is_original:
            # Stall past the straggler timeout so a retry gets submitted while this one
            # is still in flight. The retry (second call) fails immediately, so it can't
            # become a straggler itself and trigger a further resubmission.
            time.sleep(1.5)
        raise RuntimeError("cannot deepcopy usage_tracker")

    monkeypatch.setattr(parallelizer_module.copy, "deepcopy", broken_deepcopy)

    def task(item):
        return item

    executor = ParallelExecutor(num_threads=2, max_errors=10, timeout=0.3, straggler_limit=3)

    with dspy.context(usage_tracker=object()):
        results = executor.execute(task, [1])

    assert results == [None]
    assert executor.failed_indices == [0]
    assert len(executor.exceptions_map) == 1
    assert executor.error_count == 1


def test_straggler_retry_success_clears_stale_harness_failure(monkeypatch):
    """A straggler's original future can fail during harness setup while its resubmitted
    retry goes on to succeed. The success must win: results[idx] should hold the real value
    and the stale failure must not remain in failed_indices/exceptions_map alongside it.

    Ordering (harness failure recorded first, success second -- the ordering that actually
    exercises the reconciliation logic) is enforced with an Event rather than tuned sleep
    durations, so the retry's own runtime stays short and it can't become a straggler
    itself and trigger a further resubmission."""
    import dspy.utils.parallelizer as parallelizer_module

    call_count = {"n": 0}
    call_lock = threading.Lock()
    real_deepcopy = parallelizer_module.copy.deepcopy
    original_failed = threading.Event()

    def flaky_deepcopy(*args, **kwargs):
        with call_lock:
            call_count["n"] += 1
            is_original = call_count["n"] == 1
        if is_original:
            # Stall past the straggler timeout so a retry gets submitted while this one is
            # still in flight, then fail -- simulating the original attempt going bad.
            time.sleep(1.5)
            original_failed.set()
            raise RuntimeError("cannot deepcopy usage_tracker")
        # The retry: setup succeeds immediately so the task itself gets to run.
        return real_deepcopy(*args, **kwargs)

    monkeypatch.setattr(parallelizer_module.copy, "deepcopy", flaky_deepcopy)

    def task(item):
        # Don't finish until the original has failed (plus a small buffer for the main
        # thread to process it), so this success is recorded second.
        original_failed.wait(timeout=5)
        time.sleep(0.3)
        return item * 10

    executor = ParallelExecutor(num_threads=2, max_errors=10, timeout=1.0, straggler_limit=3)

    with dspy.context(usage_tracker=object()):
        results = executor.execute(task, [1])

    assert results == [10]
    assert executor.failed_indices == []
    assert executor.exceptions_map == {}


def test_straggler_retry_success_clears_cancellation_from_a_stale_failure(monkeypatch):
    """At max_errors=1, a straggler's original harness failure crosses the threshold and
    sets cancel_jobs by itself. If the resubmitted retry then succeeds, the run's only
    logical failure has been resolved -- cancel_jobs must be cleared along with error_count,
    or the caller gets a spurious cancellation for a call that actually completed cleanly.

    Same Event-based ordering as the sibling test above, just with max_errors tight enough
    that the original failure alone triggers cancellation before the retry recovers."""
    import dspy.utils.parallelizer as parallelizer_module

    call_count = {"n": 0}
    call_lock = threading.Lock()
    real_deepcopy = parallelizer_module.copy.deepcopy
    original_failed = threading.Event()

    def flaky_deepcopy(*args, **kwargs):
        with call_lock:
            call_count["n"] += 1
            is_original = call_count["n"] == 1
        if is_original:
            time.sleep(1.5)
            original_failed.set()
            raise RuntimeError("cannot deepcopy usage_tracker")
        return real_deepcopy(*args, **kwargs)

    monkeypatch.setattr(parallelizer_module.copy, "deepcopy", flaky_deepcopy)

    def task(item):
        original_failed.wait(timeout=5)
        time.sleep(0.3)
        return item * 10

    executor = ParallelExecutor(num_threads=2, max_errors=1, timeout=1.0, straggler_limit=3)

    with dspy.context(usage_tracker=object()):
        results = executor.execute(task, [1])

    assert results == [10]
    assert executor.failed_indices == []
    assert executor.exceptions_map == {}
    assert executor.error_count == 0
    assert not executor.cancel_jobs.is_set()


def test_straggler_retry_user_exception_does_not_duplicate_harness_failure(monkeypatch):
    """A straggler's original future can fail during harness setup while its resubmitted
    retry runs the task itself, which then raises. The two failures are for the same
    logical input and must not produce two failed_indices entries.

    Same Event-based ordering as the success case above, so the harness failure is
    guaranteed to be recorded before the user-function exception arrives."""
    import dspy.utils.parallelizer as parallelizer_module

    call_count = {"n": 0}
    call_lock = threading.Lock()
    real_deepcopy = parallelizer_module.copy.deepcopy
    original_failed = threading.Event()

    def flaky_deepcopy(*args, **kwargs):
        with call_lock:
            call_count["n"] += 1
            is_original = call_count["n"] == 1
        if is_original:
            time.sleep(1.5)
            original_failed.set()
            raise RuntimeError("cannot deepcopy usage_tracker")
        return real_deepcopy(*args, **kwargs)

    monkeypatch.setattr(parallelizer_module.copy, "deepcopy", flaky_deepcopy)

    def task(item):
        original_failed.wait(timeout=5)
        time.sleep(0.3)
        raise ValueError("task itself failed on the retry")

    executor = ParallelExecutor(num_threads=2, max_errors=10, timeout=1.0, straggler_limit=3)

    with dspy.context(usage_tracker=object()):
        results = executor.execute(task, [1])

    assert results == [None]
    assert executor.failed_indices == [0]
    assert len(executor.exceptions_map) == 1


def test_straggler_retry_duplicate_user_exceptions_count_once_toward_max_errors():
    """When a straggler's original attempt and its retry both raise user exceptions, the
    duplicate must not consume the error budget twice -- at max_errors=2 one logical
    failing input must not cancel the run."""
    call_count = {"n": 0}
    call_lock = threading.Lock()
    original_failed = threading.Event()

    def task(item):
        with call_lock:
            call_count["n"] += 1
            is_original = call_count["n"] == 1
        if is_original:
            time.sleep(1.5)
            original_failed.set()
            raise ValueError("original failed")
        original_failed.wait(timeout=5)
        time.sleep(0.3)
        raise ValueError("retry failed too")

    executor = ParallelExecutor(num_threads=2, max_errors=2, timeout=1.0, straggler_limit=3)

    results = executor.execute(task, [1])

    assert results == [None]
    assert executor.failed_indices == [0]
    assert executor.error_count == 1
    assert not executor.cancel_jobs.is_set()


def test_straggler_retry_recovery_is_not_abandoned_by_a_deadline():
    """Once its original attempt's failure triggers cancellation, an in-flight retry must
    be waited on until it completes -- not abandoned after a fixed window -- so a success
    that lands late still becomes the result instead of a spurious cancellation."""
    call_count = {"n": 0}
    call_lock = threading.Lock()
    original_failed = threading.Event()

    def task(item):
        with call_lock:
            call_count["n"] += 1
            is_original = call_count["n"] == 1
        if is_original:
            time.sleep(2.5)
            original_failed.set()
            raise ValueError("original failed")
        original_failed.wait(timeout=5)
        time.sleep(1.5)
        return item * 10

    executor = ParallelExecutor(num_threads=2, max_errors=1, timeout=1.0, straggler_limit=3)

    results = executor.execute(task, [1])

    assert results == [10]
    assert executor.failed_indices == []
    assert not executor.cancel_jobs.is_set()


def test_should_finalize_treats_recorded_none_result_as_already_finalized():
    """A slot can legitimately hold a real None result (not an error, not "unfinished").
    _should_finalize must tell that apart from a not-yet-recorded slot by checking against
    the _UNSET sentinel, not None -- checking against None would treat every already-None
    result as still unfinished and let a stale retry overwrite it."""
    executor = ParallelExecutor(num_threads=2)
    results = [None]

    assert executor._should_finalize(0, 99, results) is False


def test_clear_stale_failure_frees_the_error_budget():
    """A stale failure superseded by a recovered retry must give back the error-count
    budget it consumed, not just the visible failed_indices/exceptions_map entries --
    otherwise it can combine with one later unrelated failure to trigger cancellation
    even though only one input is actually still failing."""
    executor = ParallelExecutor(num_threads=2, max_errors=2)
    executor._record_error("item-a", RuntimeError("stale"))
    assert executor.error_count == 1

    executor.exceptions_map[0] = RuntimeError("stale")
    executor.failed_indices.append(0)
    executor._clear_stale_failure(0)

    assert executor.error_count == 0
    assert executor.exceptions_map == {}
    assert executor.failed_indices == []
    assert not executor.cancel_jobs.is_set()


def test_clear_stale_failure_uncancels_a_run_the_stale_failure_alone_cancelled():
    """At max_errors=1, a single stale failure sets cancel_jobs by itself. Clearing that
    failure must also clear cancel_jobs -- error_count alone dropping back under the
    threshold isn't enough if the stale cancellation is never undone."""
    executor = ParallelExecutor(num_threads=2, max_errors=1)
    executor._record_error("item-a", RuntimeError("stale"))
    assert executor.cancel_jobs.is_set()

    executor.exceptions_map[0] = RuntimeError("stale")
    executor._clear_stale_failure(0)

    assert not executor.cancel_jobs.is_set()


def test_clear_stale_failure_never_uncancels_a_real_interrupt():
    """cancel_jobs set by a genuine Ctrl-C (self.interrupted) must never be cleared by
    resolving an unrelated stale failure -- only error-budget-triggered cancellation is
    ever safe to undo."""
    executor = ParallelExecutor(num_threads=2, max_errors=1)
    executor._record_error("item-a", RuntimeError("stale"))
    executor.interrupted.set()
    assert executor.cancel_jobs.is_set()

    executor.exceptions_map[0] = RuntimeError("stale")
    executor._clear_stale_failure(0)

    assert executor.cancel_jobs.is_set()


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
