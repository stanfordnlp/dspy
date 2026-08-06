import time
from concurrent.futures import Future
from threading import Event
from unittest.mock import MagicMock

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
    worker_gate = Event()
    first_batch_started = Event()
    finish_first_batch = Event()

    class GatedUnbatchify(Unbatchify):
        def _worker(self):
            worker_gate.wait()
            super()._worker()

    def synchronized_batch_processor(batch):
        if batch == [10, 20]:
            first_batch_started.set()
            assert finish_first_batch.wait(timeout=1.0)
        return simple_batch_processor(batch)

    batch_fn_mock = MagicMock(wraps=synchronized_batch_processor)
    unbatcher = GatedUnbatchify(batch_fn=batch_fn_mock, max_batch_size=2, max_wait_time=0.5)

    futures = []
    futures.append(unbatcher.submit(10))
    assert batch_fn_mock.call_count == 0

    futures.append(unbatcher.submit(20))
    worker_gate.set()
    assert first_batch_started.wait(timeout=1.0)

    assert batch_fn_mock.call_count == 1
    batch_fn_mock.assert_called_once_with([10, 20])

    futures_3_4 = []
    futures_3_4.append(unbatcher.submit(30))
    futures_3_4.append(unbatcher.submit(40))
    finish_first_batch.set()

    results_1_2 = [f.result() for f in futures]
    results_3_4 = [f.result() for f in futures_3_4]
    assert batch_fn_mock.call_count == 2
    assert batch_fn_mock.call_args_list[1].args[0] == [30, 40]
    assert results_1_2 == [11, 21]
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
