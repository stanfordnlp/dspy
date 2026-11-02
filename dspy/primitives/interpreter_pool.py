"""An interpreter factory that starts interpreters ahead of need."""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import weakref
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreter
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.sandbox_bundle import SandboxBundleError

logger = logging.getLogger(__name__)


class InterpreterPool:
    """An ``interpreter_factory`` that keeps interpreters started ahead of need.

    Each call returns a fresh interpreter from ``factory`` that has run no code, as ``factory()`` would, except that
    it is usually already started: the pool starts ``size`` interpreters in the background and starts a replacement
    whenever it hands one out. When none is ready, a call returns a new, unstarted interpreter instead. A pooled
    ``PythonInterpreter`` can import dspy, and with ``preload_dspy`` it has already loaded it, which is what
    ``dspy.Flex`` imports first.

    Pass it wherever an ``interpreter_factory`` goes:

    ```python
    pool = dspy.InterpreterPool(size=2)
    program = dspy.Flex(MySignature, interpreter_factory=pool)
    ```

    An idle interpreter is a running sandbox process: a ``PythonInterpreter`` with dspy loaded holds about 400 MB,
    so size the pool to your memory. One interpreter hides the start-up cost when each call spends longer in LM
    calls than a sandbox takes to start; calls that run back to back, or in parallel threads, need one per call in
    flight. ``factory`` must return interchangeable interpreters, since the pool calls it ahead of time. Idle
    interpreters are shut down by ``close()``, and at exit.
    """

    def __init__(
        self,
        factory: Callable[[], CodeInterpreter] = PythonInterpreter,
        size: int = 1,
        *,
        preload_dspy: bool = True,
    ) -> None:
        if size < 1:
            raise ValueError("size must be at least 1")
        self._factory = factory
        self._preload_dspy = preload_dspy
        self._ready: queue.Queue[CodeInterpreter] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        _pools.add(self)
        for _ in range(size):
            self._refill()

    @property
    def execution_instructions(self) -> str:
        return getattr(self._factory, "execution_instructions", "") or ""

    def __call__(self) -> CodeInterpreter:
        try:
            interpreter = self._ready.get_nowait()
        except queue.Empty:
            return self._factory()
        self._refill()
        return interpreter

    def close(self) -> None:
        """Shut down the idle interpreters and stop starting new ones."""
        with self._lock:
            self._closed = True
        while True:
            try:
                interpreter = self._ready.get_nowait()
            except queue.Empty:
                return
            interpreter.shutdown()

    def __enter__(self) -> InterpreterPool:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _refill(self) -> None:
        threading.Thread(target=self._start_one, name="dspy-interpreter-pool", daemon=True).start()

    def _start_one(self) -> None:
        try:
            interpreter = self._factory()
            if isinstance(interpreter, PythonInterpreter):
                # Flex and RLM add the dspy bundle to their interpreters, which must happen before one starts.
                try:
                    interpreter.provide_dspy(preload=self._preload_dspy)
                except SandboxBundleError:
                    logger.debug("InterpreterPool starts interpreters without dspy", exc_info=True)
            interpreter.start()
        except Exception:
            # A call that finds no interpreter ready makes its own, and surfaces the failure there.
            logger.debug("InterpreterPool could not start an interpreter", exc_info=True)
            return
        with self._lock:
            if not self._closed:
                self._ready.put(interpreter)
                return
        interpreter.shutdown()


_pools: weakref.WeakSet[InterpreterPool] = weakref.WeakSet()


@atexit.register
def _close_pools() -> None:
    for pool in list(_pools):
        pool.close()
