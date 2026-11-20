"""Runtime compatibility for typed batch callables and context managers."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from dspy.utils.unbatchify import Unbatchify


def test_concurrent_scalar_calls_preserve_different_output_type():
    batches = []

    def describe_batch(values: list[int]) -> list[str]:
        batches.append(values)
        return [f"value={value}" for value in values]

    with Unbatchify(describe_batch, max_batch_size=2, max_wait_time=0.1) as worker:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(worker, 7)
            second = callers.submit(worker, 11)
            assert first.result(timeout=5) == "value=7"
            assert second.result(timeout=5) == "value=11"

    assert sorted(value for batch in batches for value in batch) == [7, 11]
    assert not worker.worker_thread.is_alive()


def test_scalar_calls_preserve_input_and_output_identity():
    input_item = {"text": "hello"}
    output_item = (5, True)

    def summarize_batch(values: list[dict[str, str]]) -> list[tuple[int, bool]]:
        assert values[0] is input_item
        return [output_item]

    with Unbatchify(summarize_batch, max_batch_size=1) as worker:
        assert worker(input_item) is output_item

    assert input_item == {"text": "hello"}


def test_batch_exception_reaches_each_caller_and_worker_can_process_next_batch():
    error = ValueError("invalid batch")

    def describe_batch(values: list[int]) -> list[str]:
        if any(value < 0 for value in values):
            raise error
        return [str(value) for value in values]

    with Unbatchify(describe_batch, max_batch_size=2, max_wait_time=0.1) as worker:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(worker, -1)
            second = callers.submit(worker, -2)
            for result in (first, second):
                with pytest.raises(ValueError, match="invalid batch") as caught:
                    result.result(timeout=5)
                assert caught.value is error

            first = callers.submit(worker, 3)
            second = callers.submit(worker, 4)
            assert first.result(timeout=5) == "3"
            assert second.result(timeout=5) == "4"


def test_context_manager_closes_worker_without_suppressing_body_exception():
    def describe_batch(values: list[int]) -> list[str]:
        return [str(value) for value in values]

    worker = Unbatchify(describe_batch, max_batch_size=1)
    with pytest.raises(RuntimeError, match="caller failed"):
        with worker as entered:
            assert entered is worker
            assert entered(5) == "5"
            raise RuntimeError("caller failed")

    assert not worker.worker_thread.is_alive()
    worker.close()
