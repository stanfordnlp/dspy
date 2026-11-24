import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy.utils.unbatchify as unbatchify_module
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


def test_unbatchify_honors_max_wait_time_under_trickling_input(monkeypatch):
    """Later arrivals must not extend the batch's timeout window."""
    clock = SimpleNamespace(now=0.0)
    arrivals = deque([0.0, 8.0, 16.0])
    ready = threading.Event()
    batches = []

    class ScheduledQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            if not block:  # Preserve get_nowait() for shutdown cleanup.
                return super().get(block=False)
            if not ready.wait(timeout=2):
                raise queue.Empty
            deadline = clock.now + timeout
            if arrivals and arrivals[0] <= deadline:
                clock.now = arrivals.popleft()
                return super().get(block=False)
            clock.now = deadline
            raise queue.Empty

    def process(batch):
        batches.append((batch, clock.now))
        return simple_batch_processor(batch)

    # Patch only this module's clock and queue, not threading's real timeouts.
    monkeypatch.setattr(unbatchify_module, "time", SimpleNamespace(time=lambda: clock.now, sleep=time.sleep))
    monkeypatch.setattr(unbatchify_module, "queue", SimpleNamespace(Queue=ScheduledQueue, Empty=queue.Empty))

    with Unbatchify(batch_fn=process, max_batch_size=10, max_wait_time=10.0) as unbatcher:
        futures = [unbatcher.submit(item) for item in (1, 2, 3)]
        ready.set()
        assert [future.result(timeout=2) for future in futures] == [2, 3, 4]

    # The arrival at 16 belongs to the next window. Re-arming the full timeout
    # would instead include it in the first batch and flush that batch at 16.
    assert batches == [([1, 2], 10.0), ([3], 20.0)]
