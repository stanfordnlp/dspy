import atexit
import logging
import threading

from dspy.primitives.python_interpreter import PythonInterpreter

logger = logging.getLogger(__name__)


class ThreadLocalInterpreter:
    """
    Manages thread-local PythonInterpreter instances for default usage.
    Ensures each thread gets its own isolated Deno process.
    """

    def __init__(self):
        self._local = threading.local()
        self._interpreters = []
        self._lock = threading.Lock()
        atexit.register(self.cleanup)

    @property
    def interpreter(self) -> PythonInterpreter:
        if not hasattr(self._local, "interpreter"):
            self._local.interpreter = PythonInterpreter()
            with self._lock:
                self._interpreters.append(self._local.interpreter)
        return self._local.interpreter

    def cleanup(self):
        with self._lock:
            for interp in self._interpreters:
                try:
                    interp.shutdown()
                except Exception:
                    pass
            self._interpreters.clear()

    def __getattr__(self, name):
        # Proxy other accesses to the underlying interpreter
        # Note: This might be tricky if attributes change dynamically, but for execute/snapshot/restore it's fine.
        return getattr(self.interpreter, name)
