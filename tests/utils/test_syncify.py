import asyncio
import gc
import warnings

import pytest

import dspy


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


def test_syncify_raises_inside_a_running_event_loop():
    """Inside a running loop the sync call fails fast toward the native async path (#10337)."""

    class MyProgram(dspy.Module):
        async def aforward(self, x: int) -> int:
            return x + 1

    sync_program = dspy.syncify(MyProgram())

    async def call_inside_loop():
        with pytest.raises(ValueError, match="native async path"):
            sync_program(1)

    asyncio.run(call_inside_loop())


def test_syncify_raise_path_does_not_leak_a_coroutine():
    """The refused coroutine is closed, so no 'never awaited' RuntimeWarning escapes."""

    class MyProgram(dspy.Module):
        async def aforward(self, x: int) -> int:
            return x + 1

    sync_program = dspy.syncify(MyProgram())

    async def call_inside_loop():
        with pytest.raises(ValueError):
            sync_program(1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        asyncio.run(call_inside_loop())
        gc.collect()


def test_syncify_and_tool_share_the_running_loop_policy():
    """syncify and Tool refuse a sync call inside a running loop the same way, so the two policies cannot drift apart again."""

    async def add_one(x: int) -> int:
        return x + 1

    tool = dspy.Tool(add_one)

    class MyProgram(dspy.Module):
        async def aforward(self, x: int) -> int:
            return x + 1

    sync_program = dspy.syncify(MyProgram())

    async def call_both():
        with pytest.raises(ValueError):
            sync_program(1)
        with dspy.context(allow_tool_async_sync_conversion=True):
            with pytest.raises(ValueError):
                tool(x=1)

    asyncio.run(call_both())
