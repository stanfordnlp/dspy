"""An interpreter factory that starts interpreters ahead of need."""

from __future__ import annotations

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
    whenever it hands one out, or after a start failed. When none is ready, a call returns a new, unstarted
    interpreter instead. A pooled ``PythonInterpreter`` has dspy installed, and usually already imported, which is
    what ``dspy.Flex`` imports first.

    Pass it wherever an ``interpreter_factory`` goes:

    ```python
    pool = dspy.InterpreterPool(size=2)
    program = dspy.Flex(MySignature, interpreter_factory=pool)
    ```

    An idle interpreter is a running sandbox process: a ``PythonInterpreter`` with dspy loaded holds about 400 MB,
    so size the pool to your memory. One interpreter hides the start-up cost when each call spends longer in LM
    calls than a sandbox takes to start; calls that run back to back, or in parallel threads, need one per call in
    flight. ``factory`` must return interchangeable interpreters, since the pool calls it ahead of time. Idle
    interpreters are shut down by ``close()``, when the pool is garbage-collected, and at exit.
    """

    def __init__(self, factory: Callable[[], CodeInterpreter] = PythonInterpreter, size: int = 1) -> None:
        if size < 1:
            raise ValueError("size must be at least 1")
        self._factory = factory
        self._size = size
        self._starting = 0
        self._ready: queue.Queue[CodeInterpreter] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        # Shuts the idle interpreters down when the pool is closed, garbage-collected, or the process exits.
        self._finalizer = weakref.finalize(self, _shut_down_idle, self._ready)
        self._refill()

    @property
    def execution_instructions(self) -> str:
        return getattr(self._factory, "execution_instructions", "") or ""

    def __call__(self) -> CodeInterpreter:
        try:
            interpreter = self._ready.get_nowait()
        except queue.Empty:
            interpreter = None
        self._refill()
        return interpreter if interpreter is not None else self._factory()

    def close(self) -> None:
        """Shut down the idle interpreters and stop starting new ones."""
        with self._lock:
            self._closed = True
        self._finalizer()

    def __enter__(self) -> InterpreterPool:
        return self

    def __copy__(self) -> InterpreterPool:
        return self  # copies of a program share its pool, as they share its LM

    def __deepcopy__(self, memo: dict) -> InterpreterPool:
        return self

    def __reduce__(self) -> tuple:
        # Saved with a program: its interpreters are processes, so loading it makes a new pool like this one.
        return type(self), (self._factory, self._size)

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _refill(self) -> None:
        """Start interpreters until ``size`` are ready or starting."""
        with self._lock:
            missing = 0 if self._closed else self._size - self._ready.qsize() - self._starting
            self._starting += max(missing, 0)
        for _ in range(missing):
            threading.Thread(target=self._start_one, name="dspy-interpreter-pool", daemon=True).start()

    def _start_one(self) -> None:
        try:
            interpreter = self._factory()
            if isinstance(interpreter, PythonInterpreter):
                # Flex and RLM add the dspy bundle to their interpreters, which must happen before one starts.
                try:
                    interpreter.provide_dspy()
                except SandboxBundleError:
                    logger.debug("InterpreterPool starts interpreters without dspy", exc_info=True)
            interpreter.start()
        except Exception:
            # A call that finds no interpreter ready makes its own, and surfaces the failure there.
            logger.debug("InterpreterPool could not start an interpreter", exc_info=True)
            with self._lock:
                self._starting -= 1
            return
        with self._lock:
            self._starting -= 1
            if not self._closed:
                self._ready.put(interpreter)
                return
        _shut_down(interpreter)


def _shut_down_idle(ready: queue.Queue[CodeInterpreter]) -> None:
    while True:
        try:
            interpreter = ready.get_nowait()
        except queue.Empty:
            return
        _shut_down(interpreter)


def _shut_down(interpreter: CodeInterpreter) -> None:
    # One that fails (its process already gone, say) does not keep the others running.
    try:
        interpreter.shutdown()
    except Exception:
        logger.debug("InterpreterPool could not shut down an idle interpreter", exc_info=True)
