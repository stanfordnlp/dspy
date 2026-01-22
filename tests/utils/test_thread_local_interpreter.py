
import shutil
import threading

import pytest

from dspy.utils.interpreter import ThreadLocalInterpreter

# This test suite requires deno to be installed.
is_deno_available = shutil.which("deno") is not None

@pytest.mark.skipif(not is_deno_available, reason="Deno is not installed or not in PATH")
def test_thread_local_interpreter_cleanup():
    """Verify that ThreadLocalInterpreter.cleanup() terminates all managed processes."""

    tli = ThreadLocalInterpreter()

    # Initialize an interpreter in the main thread
    main_interp = tli.interpreter
    main_interp.execute("print('main')")
    assert main_interp.deno_process is not None
    assert main_interp.deno_process.poll() is None

    # Initialize an interpreter in a separate thread
    thread_interp = [None]
    def run_in_thread():
        tli.interpreter.execute("print('thread')")
        thread_interp[0] = tli.interpreter

    t = threading.Thread(target=run_in_thread)
    t.start()
    t.join()

    # Verify we have two distinct interpreters
    child_interp = thread_interp[0]
    assert child_interp is not None
    assert child_interp is not main_interp
    assert child_interp.deno_process is not None
    assert child_interp.deno_process.poll() is None

    # Run cleanup
    tli.cleanup()

    # Verify both processes are terminated
    assert main_interp.deno_process is None
    assert child_interp.deno_process is None # ThreadLocal cleanup might clear the list logic

    # Check underlying assumption if references were kept
    # Note: ThreadLocalInterpreter.cleanup calls shutdown() which sets deno_process to None
    # We can't check poll() if it's None, so checking it's None is the correct assertion
    # for the current implementation of PythonInterpreter.shutdown().

def test_thread_local_interpreter_call_proxy():
    """Verify that ThreadLocalInterpreter instances are callable (proxying to execute)."""
    tli = ThreadLocalInterpreter()

    # Test __call__
    result = tli("1+1")
    # Output might be int or string depending on interpreter version/mock, usually string "2"
    assert str(result).strip() == "2"

    # Test getattr proxy
    assert tli.execute == tli.interpreter.execute
