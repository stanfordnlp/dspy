"""Reusable DSPy program execution helpers."""

from __future__ import annotations

import asyncio
import sys
import threading
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
    """Sync-callable wrapper that drives ``program.acall`` per thread."""

    def __init__(self, program: dspy.Module) -> None:
        self.program = program
        self._tls = threading.local()
        self._loops: list[asyncio.AbstractEventLoop] = []
        self._loops_lock = threading.Lock()

    def _loop(self) -> asyncio.AbstractEventLoop:
        loop = getattr(self._tls, "loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            self._tls.loop = loop
            with self._loops_lock:
                self._loops.append(loop)
        return loop

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        loop = self._loop()
        return loop.run_until_complete(self.program.acall(*args, **kwargs))

    def close(self) -> None:
        """Close every per-thread loop created by this runner."""
        with self._loops_lock:
            loops = list(self._loops)
            self._loops.clear()
        for loop in loops:
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
