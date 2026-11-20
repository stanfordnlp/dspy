"""Static contracts for scalar calls backed by a typed batch processor."""

from collections.abc import Callable

from typing_extensions import assert_type

from dspy.utils.unbatchify import Unbatchify


def describe_batch(values: list[int]) -> list[str]:
    return [str(value) for value in values]


def check_inferred_types() -> None:
    worker = Unbatchify(describe_batch)
    assert_type(worker(3), str)
    worker("three")  # ty: ignore[invalid-argument-type]
    worker.close()


def check_context_manager() -> None:
    with Unbatchify(describe_batch, max_batch_size=2, max_wait_time=0.01) as worker:
        assert_type(worker(3), str)
        worker("three")  # ty: ignore[invalid-argument-type]


def check_explicit_types(batch_fn: Callable[[list[int]], list[str]]) -> None:
    worker: Unbatchify[int, str] = Unbatchify(batch_fn)
    assert_type(worker(1), str)
    worker.close()


def check_structured_outputs() -> None:
    def summarize_batch(values: list[str]) -> list[tuple[int, bool]]:
        return [(len(value), bool(value)) for value in values]

    with Unbatchify(summarize_batch) as worker:
        assert_type(worker("hello"), tuple[int, bool])
        worker(3)  # ty: ignore[invalid-argument-type]
