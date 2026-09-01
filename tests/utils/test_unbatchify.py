import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from dspy.utils.unbatchify import Unbatchify


def simple_batch_processor(batch):
    """A simple batch function that adds 1 to each item."""
    return [item + 1 for item in batch]


def submit(self, input_item: any) -> Future:
    """Submits an item for processing and returns a Future."""
    future = Future()
    self.input_queue.put((input_item, future))
    return future


Unbatchify.submit = submit


def test_unbatchify_batch_size_trigger():
    """Test that the batch processes exactly when max_batch_size is reached."""
    batch_fn_mock = MagicMock(wraps=simple_batch_processor)
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=2, max_wait_time=5.0)

    futures = []
    futures.append(unbatcher.submit(10))
    time.sleep(0.02)
    assert batch_fn_mock.call_count == 0

    futures.append(unbatcher.submit(20))

    results_1_2 = [f.result() for f in futures]
    assert batch_fn_mock.call_count == 1
    batch_fn_mock.assert_called_once_with([10, 20])
    assert results_1_2 == [11, 21]

    futures_3_4 = []
    futures_3_4.append(unbatcher.submit(30))
    futures_3_4.append(unbatcher.submit(40))

    results_3_4 = [f.result() for f in futures_3_4]
    time.sleep(0.01)
    assert batch_fn_mock.call_count == 2
    assert batch_fn_mock.call_args_list[1].args[0] == [30, 40]
    assert results_3_4 == [31, 41]

    unbatcher.close()


def test_unbatchify_timeout_trigger():
    """Test that the batch processes after max_wait_time."""
    batch_fn_mock = MagicMock(wraps=simple_batch_processor)
    wait_time = 0.15
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=5, max_wait_time=wait_time)

    futures = []
    futures.append(unbatcher.submit(100))
    futures.append(unbatcher.submit(200))

    time.sleep(wait_time / 2)
    assert batch_fn_mock.call_count == 0

    results = [f.result() for f in futures]

    assert batch_fn_mock.call_count == 1
    batch_fn_mock.assert_called_once_with([100, 200])
    assert results == [101, 201]

    unbatcher.close()


def test_unbatchify_honors_max_wait_time_under_trickling_input():
    """A batch must flush within max_wait_time of the FIRST item arriving, even if
    later items keep trickling in just under max_wait_time apart from each other.
    Each queue.get() must wait only the time remaining in the window, not the full
    max_wait_time again - otherwise the total wait can be a multiple of the budget."""
    batch_fn_mock = MagicMock(wraps=simple_batch_processor)
    wait_time = 0.1
    unbatcher = Unbatchify(batch_fn=batch_fn_mock, max_batch_size=10, max_wait_time=wait_time)

    start = time.time()
    future_1 = unbatcher.submit(1)
    time.sleep(wait_time * 0.8)  # arrives well within the window, but close to its edge
    future_2 = unbatcher.submit(2)

    results = [future_1.result(timeout=2), future_2.result(timeout=2)]
    elapsed = time.time() - start

    assert results == [2, 3]
    # Generous slack for scheduling jitter, but must stay well under 2x the budget
    # (which is what the unfixed code produces for this timing).
    assert elapsed < wait_time * 1.5

    unbatcher.close()


def test_unbatchify_raises_when_batch_fn_returns_fewer_outputs():
    """Every caller in the batch must fail fast if batch_fn drops outputs, instead of hanging."""

    def dropping_batch_fn(batch):
        # Simulates a batch_fn that filters its inputs, e.g. skipping items it cannot handle.
        return [item for item in batch if item % 2 == 0]

    unbatcher = Unbatchify(batch_fn=dropping_batch_fn, max_batch_size=2, max_wait_time=5.0)

    try:
        futures = [unbatcher.submit(1), unbatcher.submit(2)]

        for future in futures:
            with pytest.raises(ValueError, match="exactly one output per input"):
                future.result(timeout=5)
    finally:
        unbatcher.close()


def test_unbatchify_raises_when_batch_fn_returns_more_outputs():
    """Extra outputs are a contract violation too, and must not be silently discarded."""

    def duplicating_batch_fn(batch):
        return [item for item in batch for _ in range(2)]

    unbatcher = Unbatchify(batch_fn=duplicating_batch_fn, max_batch_size=2, max_wait_time=5.0)

    try:
        futures = [unbatcher.submit(1), unbatcher.submit(2)]

        for future in futures:
            with pytest.raises(ValueError, match="exactly one output per input"):
                future.result(timeout=5)
    finally:
        unbatcher.close()


def test_unbatchify_does_not_misalign_outputs_when_batch_fn_drops_items():
    """A dropped output must never shift a later output onto an earlier input's future."""

    def drop_first_batch_fn(batch):
        return [f"result-for-{item}" for item in batch[1:]]

    unbatcher = Unbatchify(batch_fn=drop_first_batch_fn, max_batch_size=2, max_wait_time=5.0)

    try:
        future = unbatcher.submit("a")
        unbatcher.submit("b")

        # Before the fix this future resolved to "result-for-b" -- the wrong input's output.
        with pytest.raises(ValueError, match="exactly one output per input"):
            future.result(timeout=5)
    finally:
        unbatcher.close()
