import multiprocessing
from typing import Optional


class PythonREPL:
    @classmethod
    def worker(cls, code: str, globals, locals, queue):
        try:
            exec(code, globals, locals)
            queue.put(None)
        except Exception as e:
            queue.put(repr(e))

    def run(self, command: str, timeout=5, globals={}, locals=None) -> Optional[str]:
        queue: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=self.worker, args=(command, globals, locals, queue)
        )
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            return "TimeoutError"
        return queue.get_nowait()


# We had to move the code above to a class for pickle reasons
def execute_code(command: str, timeout=5, globals={}, locals=None) -> Optional[str]:
    return PythonREPL().run(command, timeout, globals, locals)
