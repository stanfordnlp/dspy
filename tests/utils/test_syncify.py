import asyncio
import threading

import dspy
from dspy.utils.metric_call import call_metric


def test_syncify_in_place():
    class MyProgram(dspy.Module):
        async def aforward(self, x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

    sync_program = dspy.syncify(MyProgram())
    assert sync_program(1) == 2
    assert sync_program(2) == 3


def test_syncify_with_wrapper():
    class MyProgram(dspy.Module):
        async def aforward(self, x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

    sync_program = dspy.syncify(MyProgram(), in_place=False)
    assert sync_program(1) == 2
    assert sync_program(2) == 3


def test_syncify_works_with_optimizers():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question->answer")

        async def aforward(self, question: str):
            return await self.predict.acall(question=question)

    async_program = MyProgram()

    def dummy_metric(gold, pred, traces=None):
        return True

    # We only test the optimizer completes without errors, so the LM response doesn't matter.
    lm = dspy.utils.DummyLM([{"answer": "dummy"} for _ in range(100)])
    dspy.configure(lm=lm)

    dataset = [dspy.Example(question="question", answer="answer").with_inputs("question") for _ in range(10)]

    optimizer = dspy.BootstrapFewShot(metric=dummy_metric, max_bootstrapped_demos=2, max_labeled_demos=0)

    # Test syncify in place
    sync_program = dspy.syncify(async_program, in_place=True)
    optimized_program = optimizer.compile(sync_program, trainset=dataset)
    assert len(optimized_program.predictors()[0].demos) == 2

    # Test syncify with wrapper
    sync_program = dspy.syncify(async_program, in_place=False)
    optimized_program = optimizer.compile(sync_program, trainset=dataset)
    assert len(optimized_program.predictors()[0].demos) == 2


def test_call_metric_sync_without_running_loop():
    assert call_metric(lambda x: x + 1, 1) == 2


def test_call_metric_async_without_running_loop():
    async def metric(x):
        await asyncio.sleep(0)
        return x + 1

    assert call_metric(metric, 1) == 2


def test_call_metric_with_running_loop_uses_thread_without_running_loop():
    threads = []

    def metric(x):
        threads.append(threading.current_thread())
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return x + 1
        raise AssertionError("metric ran on a thread with a running event loop")

    async def main():
        return call_metric(metric, 1)

    assert asyncio.run(main()) == 2
    assert threads and threads[0] is not threading.main_thread()


def test_call_metric_async_with_running_loop():
    async def metric(x):
        await asyncio.sleep(0)
        return x + 1

    async def main():
        return call_metric(metric, 1)

    assert asyncio.run(main()) == 2


def test_call_metric_propagates_settings_to_worker_thread():
    with dspy.context(lm="sentinel"):

        def metric():
            assert dspy.settings.lm == "sentinel"
            return True

        async def main():
            return call_metric(metric)

        assert asyncio.run(main())
