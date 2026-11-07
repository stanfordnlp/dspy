"""Callback decorators preserve caller-visible sync and async contracts.

Deliberately invalid calls use ignores to assert expected diagnostics. Unused-ignore
errors make these checks fail if the decorators stop rejecting invalid calls.
"""

import asyncio
from typing import Any, Coroutine, TypeVar, overload

from typing_extensions import assert_type

from dspy import Evaluate, Module
from dspy.evaluate import EvaluationResult
from dspy.utils.callback import with_callbacks
from dspy.utils.callback_context import _bind_active_call_id

T = TypeVar("T")


class Runner:
    @with_callbacks
    def run(self, count: int, /, *, prefix: str) -> str:
        return f"{prefix}: {count}"

    @with_callbacks
    async def arun(self, count: int, /, *, prefix: str) -> int:
        return count + len(prefix)

    @with_callbacks
    def deferred(self, count: int) -> Coroutine[Any, Any, int]:
        return self.arun(count, prefix="later")

    @with_callbacks
    def identity(self, value: T) -> T:
        return value

    @overload
    def echo(self, value: int) -> int: ...

    @overload
    def echo(self, value: str) -> str: ...

    def echo(self, value: int | str) -> int | str:
        return value


class UnnamedCallable:
    def __call__(self, count: int) -> int:
        return count


async def check_callbacks(evaluate: Evaluate, program: Module, value: str) -> None:
    runner = Runner()
    assert_type(runner.run(3, prefix="count"), str)
    assert_type(Runner.run(runner, 3, prefix="count"), str)
    assert_type(runner.identity(value), str)
    assert_type(await runner.arun(3, prefix="count"), int)
    assert_type(await asyncio.create_task(runner.arun(3, prefix="count")), int)
    assert_type(await runner.deferred(3), int)
    assert_type(evaluate(program), EvaluationResult)

    runner.run("three", prefix="count")  # ty: ignore[invalid-argument-type]
    runner.run(3)  # ty: ignore[missing-argument]
    runner.run(3, "count", prefix="count")  # ty: ignore[too-many-positional-arguments]
    runner.run(count=3, prefix="count")  # ty: ignore[positional-only-parameter-as-kwarg]
    await runner.arun("three", prefix="count")  # ty: ignore[invalid-argument-type]
    await runner.arun(3)  # ty: ignore[missing-argument]

    bound = _bind_active_call_id(runner.run)
    assert_type(bound(3, prefix="bound"), str)
    bound("three", prefix="bound")  # ty: ignore[invalid-argument-type]

    overloaded = with_callbacks(Runner.echo)
    assert_type(overloaded(runner, 3), int)
    assert_type(overloaded(runner, "three"), str)
    overloaded(runner, [])  # ty: ignore[no-matching-overload]

    with_callbacks(UnnamedCallable())  # ty: ignore[invalid-argument-type]
