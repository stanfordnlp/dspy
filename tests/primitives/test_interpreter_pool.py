"""InterpreterPool: an interpreter factory that starts interpreters ahead of need."""

import shutil
import textwrap
import threading
import time

import pytest

import dspy
from dspy.primitives.code_interpreter import _validate_interpreter_factory
from dspy.utils.dummies import DummyLM


class Recorder:
    """A factory of LocalInterpreters that records each one it makes and whether it was started."""

    execution_instructions = dspy.LocalInterpreter.execution_instructions

    def __init__(self):
        self.made = []

    def __call__(self):
        interpreter = dspy.LocalInterpreter()
        self.made.append(interpreter)
        return interpreter


def _wait_for(condition, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_pool_is_a_valid_factory_and_rejects_an_empty_size():
    pool = dspy.InterpreterPool(Recorder(), size=1)
    try:
        _validate_interpreter_factory(pool)
        assert pool.execution_instructions == dspy.LocalInterpreter.execution_instructions
    finally:
        pool.close()
    with pytest.raises(ValueError):
        dspy.InterpreterPool(Recorder(), size=0)


def test_pool_hands_out_each_started_interpreter_once_and_refills():
    factory = Recorder()
    with dspy.InterpreterPool(factory, size=2) as pool:
        _wait_for(lambda: pool._ready.qsize() == 2)
        first, second = pool(), pool()
        assert first is not second
        assert first._process is not None and second._process is not None  # started before they were asked for
        _wait_for(lambda: pool._ready.qsize() == 2)  # each one handed out was replaced
        assert len(factory.made) == 4
        assert first.execute("1 + 1") == 2
        first.shutdown()
        second.shutdown()
    assert all(interpreter._process is None for interpreter in factory.made)  # close() stopped the idle ones


def test_pool_falls_back_to_a_new_interpreter_when_none_is_ready():
    gate = threading.Event()

    class SlowRefills(Recorder):
        def __call__(self):
            if self.made and threading.current_thread().name == "dspy-interpreter-pool":
                gate.wait()  # the replacement is still starting
            return super().__call__()

    factory = SlowRefills()
    pool = dspy.InterpreterPool(factory, size=1)
    try:
        _wait_for(lambda: pool._ready.qsize() == 1)
        started = pool()
        cold = pool()
        assert started._process is not None
        assert cold is factory.made[-1] and cold._process is None  # made on the spot, not started
        assert cold.execute("2 * 3") == 6
        for interpreter in (started, cold):
            interpreter.shutdown()
    finally:
        gate.set()
        pool.close()


@pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")
@pytest.mark.parametrize("preload_dspy", [True, False])
def test_pooled_python_interpreters_can_import_dspy_with_or_without_preloading(preload_dspy):
    class Doubler(dspy.Signature):
        value: int = dspy.InputField()
        result: int = dspy.OutputField()

    with dspy.InterpreterPool(size=1, preload_dspy=preload_dspy) as pool:
        _wait_for(lambda: pool._ready.qsize() == 1)
        flex = dspy.Flex(Doubler, interpreter_factory=pool)
        with dspy.context(lm=DummyLM([{"result": "4"}])):
            assert flex(value=2).result == 4


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(dspy.LocalInterpreter, id="local"),
        pytest.param(
            dspy.PythonInterpreter,
            marks=pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed"),
            id="deno",
        ),
    ],
)
def test_flex_runs_on_pooled_interpreters(factory):
    class Doubler(dspy.Signature):
        value: int = dspy.InputField()
        result: int = dspy.OutputField()

    with dspy.InterpreterPool(factory, size=2) as pool:
        flex = dspy.Flex(Doubler, interpreter_factory=pool)
        flex._bind_code(textwrap.dedent(
            """
            class M(dspy.Module):
                def forward(self, value):
                    inner = dspy.RLM("value: int -> result: int", max_iters=1)
                    return dspy.Prediction(result=inner(value=value).result + self.calls())

                def calls(self):
                    import builtins
                    builtins.calls = getattr(builtins, "calls", 0) + 1
                    return builtins.calls - 1
            """
        ).strip())
        with dspy.context(lm=DummyLM([{"plan": "p", "code": "SUBMIT(result=value * 2)"}] * 4)):
            # A pooled interpreter is fresh: state from the first forward does not reach the second.
            assert [flex(value=2).result, flex(value=3).result] == [4, 6]
