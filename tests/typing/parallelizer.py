"""Executor contracts checked by ty, not executed by pytest.

Ignores on deliberately invalid calls assert expected diagnostics: unused-ignore
errors make these checks fail if invalid calls become accepted.
"""

from typing_extensions import assert_type

from dspy.utils.parallelizer import ParallelExecutor


def format_count(count: int) -> str:
    return str(count)


def measure(value: str) -> float:
    return len(value) / 2


def optional_result(count: int) -> str | None:
    return str(count) if count else None


def check_executor() -> None:
    executor = ParallelExecutor(num_threads=3)
    assert_type(executor.execute(format_count, range(3)), list[str | None])
    assert_type(executor.execute(measure, ("one", "three")), list[float | None])
    assert_type(executor.execute(optional_result, [0, 1]), list[str | None])
    assert_type(executor.failed_indices, list[int])
    assert_type(executor.exceptions_map, dict[int, Exception])

    sequential = ParallelExecutor(num_threads=1)
    assert_type(sequential.execute(format_count, {1, 3}), list[str | None])

    executor.execute(format_count, ["one"])  # ty: ignore[invalid-argument-type]
    executor.execute(format_count, iter([1, 3]))  # ty: ignore[invalid-argument-type]
    ParallelExecutor(num_threads="three")  # ty: ignore[invalid-argument-type]
