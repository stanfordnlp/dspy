"""Reusable DSPy program execution helpers."""

from __future__ import annotations

import asyncio
import sys
import traceback
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt

import dspy

__all__ = [
    "AsyncProgramRunner",
    "PredictorDemoCounts",
    "count_predictor_demos",
    "evaluate_program_with_async_runner",
]


class PredictorDemoCounts(BaseModel):
    """Demo counts for a compiled DSPy module."""

    model_config = ConfigDict(extra="forbid")

    total: StrictInt
    by_predictor: list[tuple[str, int]]


class AsyncProgramRunner:
    """Drive ``program.acall`` from sync evaluators."""

    def __init__(self, program: dspy.Module) -> None:
        self.program = program

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.program.acall(*args, **kwargs))
        finally:
            self._close_loop(loop)

    def close(self) -> None:
        """Kept for the evaluator fallback path."""

    @staticmethod
    def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e:
            print(
                f"[AsyncProgramRunner shutdown_asyncgens] {e!r}",
                file=sys.stderr,
            )
        try:
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception as e:
            print(
                f"[AsyncProgramRunner shutdown_default_executor] {e!r}",
                file=sys.stderr,
            )
        try:
            if not loop.is_closed():
                loop.close()
        except Exception as e:
            print(f"[AsyncProgramRunner close] {e!r}", file=sys.stderr)


def count_predictor_demos(compiled: dspy.Module) -> PredictorDemoCounts:
    """Count demos attached to each predictor in a compiled DSPy module."""
    total = 0
    by_predictor: list[tuple[str, int]] = []
    for name, predictor in compiled.named_predictors():
        count = len(getattr(predictor, "demos", []) or [])
        by_predictor.append((name, count))
        total += count
    return PredictorDemoCounts(total=total, by_predictor=by_predictor)


def evaluate_program_with_async_runner(
    evaluator: dspy.Evaluate, program: dspy.Module
) -> Any:
    """Evaluate with async DSPy calls, falling back to direct calls."""
    runner = AsyncProgramRunner(program)
    try:
        return evaluator(runner)
    except Exception:
        traceback.print_exc()
        return evaluator(program)
    finally:
        runner.close()
