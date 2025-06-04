import threading
import dspy
from dspy.utils.dummies import DummyLM
from dspy.utils.parallelizer import ParallelExecutor


def test_parallelexecutor_propagates_context():
    lm = DummyLM([{"output": "x"}])
    dspy.settings.configure(X=lm)

    def worker(idx):
        assert dspy.settings.X == lm
        for step in range(0, 50, 11):
            with dspy.settings.context(X=idx * step):
                assert dspy.settings.X == idx * step
            assert dspy.settings.X == lm
        return idx

    executor = ParallelExecutor(num_threads=4, max_errors=1)
    results = executor.execute(worker, list(range(5)))

    assert results == list(range(5))
    assert dspy.settings.X == lm


def test_manual_threads_do_not_inherit_context():
    dspy.settings.configure(X="base")
    results = []

    with dspy.settings.context(Y=1):
        def worker(idx):
            assert dspy.settings.X == "base"
            assert not hasattr(dspy.settings, "Y")
            results.append(idx)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == [0, 1, 2]
        assert dspy.settings.Y == 1

    assert not hasattr(dspy.settings, "Y")
    assert dspy.settings.X == "base"
