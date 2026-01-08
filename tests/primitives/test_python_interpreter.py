import threading

import pytest

from dspy.primitives.python_interpreter import PythonInterpreter


@pytest.fixture
def interpreter():
    interpreter = PythonInterpreter()
    yield interpreter
    interpreter.shutdown()


def test_variable_isolation(interpreter):
    """Test that variables are cleared after restore_state."""
    # Baseline snapshot
    interpreter.snapshot_state()

    # Define variable
    interpreter.execute("x = 100")
    assert int(interpreter.execute("x")) == 100

    # Restore
    interpreter.restore_state()

    # Verify x is gone
    result = interpreter.execute("try:\n    print(x)\nexcept NameError:\n    print('NameError')")
    assert "NameError" in result or "not defined" in str(result)


def test_module_unloading(interpreter):
    """Test that imported modules are unloaded after restore_state."""
    interpreter.snapshot_state()

    # Import module
    interpreter.execute("import math")
    assert float(interpreter.execute("math.pi")) > 3.0

    # Restore
    interpreter.restore_state()

    # Verify math is gone
    result = interpreter.execute("try:\n    print(math.pi)\nexcept NameError:\n    print('NameError')")
    assert "NameError" in result


def test_environment_variable_reversion(interpreter):
    """Test that environment variables are reverted after restore_state."""
    interpreter.snapshot_state()

    # Set env var
    interpreter.execute("import os; os.environ['TEST_ENV_VAR'] = 'dspy_test'")
    assert interpreter.execute("import os; os.environ.get('TEST_ENV_VAR')") == "dspy_test"

    # Restore
    interpreter.restore_state()

    # Verify env var is gone
    # Note: runner.js returns empty string or None for missing env vars or None print result
    result = interpreter.execute("import os; os.environ.get('TEST_ENV_VAR')")
    assert result is None or result == "None" or result == ""


def test_thread_safety_locking(interpreter):
    """Test that multiple threads can share the interpreter without crashing."""

    def worker(tid, results):
        try:
            # Execute something that takes a tiny bit of time to encourage overlap
            res = interpreter.execute(f"import time; time.sleep(0.01); 'thread_{tid}'")
            results[tid] = res
        except Exception as e:
            results[tid] = e

    threads = []
    results = {}
    for i in range(5):
        t = threading.Thread(target=worker, args=(i, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for i in range(5):
        assert results[i] == f"thread_{i}", f"Thread {i} failed: {results[i]}"
