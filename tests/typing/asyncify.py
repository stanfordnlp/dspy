"""Static contracts, checked by ty rather than executed by pytest.

Unused-ignore errors make the deliberately invalid calls fail if type safety is lost.
"""

import asyncio

from typing_extensions import assert_type

from dspy import asyncify


def format_count(count: int, /, *, prefix: str) -> str:
    return f"{prefix}: {count}"


async def check_asyncify() -> None:
    wrapped = asyncify(format_count)
    assert_type(await wrapped(3, prefix="count"), str)
    assert_type(await asyncio.create_task(wrapped(3, prefix="count")), str)

    await wrapped("three", prefix="count")  # ty: ignore[invalid-argument-type]
    await wrapped(3)  # ty: ignore[missing-argument]
    await wrapped(3, "count", prefix="count")  # ty: ignore[too-many-positional-arguments]
    await wrapped(count=3, prefix="count")  # ty: ignore[positional-only-parameter-as-kwarg]
