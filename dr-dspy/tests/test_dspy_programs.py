from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dr_dspy.dspy_programs import AsyncProgramRunner


class RecordingLoop:
    def __init__(self) -> None:
        self.created_thread = threading.get_ident()
        self.closed_thread: int | None = None
        self._closed = False

    def run_until_complete(self, awaitable: Any) -> Any:
        return asyncio.run(awaitable)

    async def shutdown_asyncgens(self) -> None:
        return None

    async def shutdown_default_executor(self) -> None:
        return None

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self.closed_thread = threading.get_ident()
        self._closed = True


class AsyncProgram:
    async def acall(self) -> int:
        return threading.get_ident()


def test_async_program_runner_closes_loop_in_worker_thread(monkeypatch) -> None:
    loops: list[RecordingLoop] = []

    def new_event_loop() -> RecordingLoop:
        loop = RecordingLoop()
        loops.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", new_event_loop)
    runner = AsyncProgramRunner(AsyncProgram())

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker_thread = executor.submit(runner).result()

    assert len(loops) == 1
    assert loops[0].created_thread == worker_thread
    assert loops[0].closed_thread == worker_thread
