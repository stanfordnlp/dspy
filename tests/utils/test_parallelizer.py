import pytest

from dspy.utils.parallelizer import ParallelExecutor


def test_single_thread():
    data = [0, 1, 2, 3, 4]
    executor = ParallelExecutor(num_threads=1)
    assert executor.execute(lambda x: x, data) == data


def test_failing_function():
    """
    If any item raises an exception, ParallelExecutor should cancel execution
    and raise an exception in the main thread without hanging.
    """
    data = [0, 1, "boom", 3, 4]

    def failing_func(x):
        if x == "boom":
            raise ValueError("Simulated error!")
        return x

    executor = ParallelExecutor(
        num_threads=2,
        max_errors=1,  # Immediately cancel after the first error
        provide_traceback=True,
    )

    with pytest.raises(ValueError, match="Simulated error!"):
        _ = executor.execute(failing_func, data)


def test_max_errors():
    """
    If the number of errors exceeds max_errors, the execution should be cancelled.
    """
    data = [0, 1, "boom1", "boom2", "boom3", "boom4", 3, 4]

    def failing_func(x):
        if isinstance(x, str) and x.startswith("boom"):
            raise ValueError(f"Simulated error! {x}")
        return x

    executor = ParallelExecutor(
        num_threads=2,
        max_errors=4,
        provide_traceback=True,
    )

    with pytest.raises(ValueError, match="Simulated error! boom4"):
        _ = executor.execute(failing_func, data)


def test_sigint_interrupt():
    """
    Demonstrate a synthetic Ctrl+C that cancels execution mid-stream.
    In practice, you might just press Ctrl+C manually while running pytest.
    """
    import time
    import signal

    data = [0, 1, 2, 3, 4]

    def interrupting_func(x):
        if x == 2:
            # Simulate some work
            time.sleep(0.2)
            # Then simulate hitting Ctrl+C
            signal.raise_signal(signal.SIGINT)
        time.sleep(0.1)
        return x

    executor = ParallelExecutor(
        num_threads=2,
        max_errors=5,
        provide_traceback=True,
    )

    # We expect a cancellation when 2 is processed
    with pytest.raises(KeyboardInterrupt):
        _ = executor.execute(interrupting_func, data)
