"""Signal dispositions are values as well as callables."""

import signal

import pytest

from dspy.utils import parallelizer


@pytest.mark.parametrize("original", [signal.SIG_DFL, signal.SIG_IGN])
def test_interrupt_cancels_and_restores_sentinel_handler(monkeypatch, original):
    installed = []
    monkeypatch.setattr(parallelizer.signal, "getsignal", lambda signum: original)
    monkeypatch.setattr(parallelizer.signal, "signal", lambda signum, handler: installed.append(handler))

    def interrupt(*args, **kwargs):
        installed[-1](signal.SIGINT, None)

    monkeypatch.setattr(parallelizer, "wait", interrupt)
    executor = parallelizer.ParallelExecutor(num_threads=2, disable_progress_bar=True)
    with pytest.raises(KeyboardInterrupt):
        executor.execute(lambda value: value, [1, 2])
    assert executor.cancel_jobs.is_set()
    assert installed[-1] is original
