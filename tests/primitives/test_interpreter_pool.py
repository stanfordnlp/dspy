"""InterpreterPool: an interpreter factory that starts interpreters ahead of need."""

import copy
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


def test_a_dropped_pool_shuts_down_its_idle_interpreters():
    import gc

    factory = Recorder()
    pool = dspy.InterpreterPool(factory, size=1)
    ready = pool._ready
    _wait_for(lambda: ready.qsize() == 1)
    del pool
    gc.collect()
    assert [interpreter._process for interpreter in factory.made] == [None]


def test_closing_shuts_down_every_idle_interpreter_even_if_one_fails():
    factory = Recorder()
    pool = dspy.InterpreterPool(factory, size=2)
    _wait_for(lambda: pool._ready.qsize() == 2)
    for interpreter in factory.made:  # each shuts down, then raises, as when its process is already gone

        def broken_shutdown(real=interpreter.shutdown):
            real()
            raise BrokenPipeError

        interpreter.shutdown = broken_shutdown
    pool.close()
    assert [interpreter._process for interpreter in factory.made] == [None, None]


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


def test_pool_starts_again_after_its_background_starts_failed():
    class FailsAtFirst(Recorder):
        def __init__(self):
            super().__init__()
            self.failures, self.lock = 2, threading.Lock()

        def __call__(self):
            with self.lock:
                if self.failures:
                    self.failures -= 1
                    raise RuntimeError("not yet")
            return super().__call__()

    with dspy.InterpreterPool(FailsAtFirst(), size=2) as pool:
        _wait_for(lambda: pool._starting == 0)  # both first starts failed
        assert pool._ready.qsize() == 0
        cold = pool()
        _wait_for(lambda: pool._ready.qsize() == 2)  # the call started replacements for them
        cold.shutdown()


@pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")
def test_failed_dspy_warmup_does_not_poison_the_next_flex_call(monkeypatch):
    from dspy.primitives import sandbox_bundle

    def fail_preparation(*args, **kwargs):
        raise sandbox_bundle.SandboxBundleError("transient bundle failure")

    with monkeypatch.context() as warmup:
        warmup.setattr(sandbox_bundle, "prepare_bundle", fail_preparation)
        with dspy.InterpreterPool(size=1) as pool:
            _wait_for(lambda: pool._starting == 0)
            warmup.undo()  # Preparation recovers before the public invocation, not before pool startup.
            flex = dspy.Flex("value: int -> result: int", interpreter_factory=pool)
            with dspy.context(lm=DummyLM([{"result": "4"}, {"result": "6"}])):
                assert flex(value=2).result == 4
                _wait_for(lambda: pool._ready.qsize() == 1)
                assert flex(value=3).result == 6


@pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")
def test_pooled_python_interpreters_can_import_dspy():
    class Doubler(dspy.Signature):
        value: int = dspy.InputField()
        result: int = dspy.OutputField()

    with dspy.InterpreterPool(size=1) as pool:
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


def test_a_program_with_a_pool_can_be_copied_and_saved(tmp_path):
    with dspy.InterpreterPool(dspy.LocalInterpreter, size=1) as pool:
        flex = dspy.Flex("question -> answer", interpreter_factory=pool)
        assert copy.deepcopy(flex)._interpreter_factory is pool  # copies share the pool
        flex.save(tmp_path / "program", save_program=True)
        restored = dspy.load(tmp_path / "program", allow_pickle=True)._interpreter_factory
        try:
            assert isinstance(restored, dspy.InterpreterPool) and restored is not pool  # a new pool like it
            assert (restored._factory, restored._size) == (dspy.LocalInterpreter, 1)
        finally:
            restored.close()


class Provided:
    """A factory of interpreters that record the libraries each was given before it started."""

    def __init__(self):
        self.made = []

    def __call__(self):
        interpreter = dspy.LocalInterpreter()
        interpreter.given = None
        interpreter.provide_libraries = lambda modules, interpreter=interpreter: setattr(interpreter, "given", tuple(modules))
        self.made.append(interpreter)
        return interpreter


def test_a_pool_starts_its_interpreters_with_the_libraries_it_is_asked_for():
    factory = Provided()
    with dspy.InterpreterPool(factory, size=2) as pool:
        _wait_for(lambda: pool._ready.qsize() == 2)
        before = list(factory.made)
        assert [interpreter.given for interpreter in before] == [None, None]

        pool.provide_libraries(["shop"])  # as a dspy.Flex with libraries does before it takes an interpreter

        assert all(interpreter._process is None for interpreter in before)  # those without it were shut down
        _wait_for(lambda: pool._ready.qsize() == 2)
        first = pool()
        assert first.given == ("shop",) and first._process is not None
        first.shutdown()

        made = len(factory.made)
        pool.provide_libraries(["shop"])  # nothing new: nothing is replaced
        assert len(factory.made) == made
        pool.provide_libraries(["tax", "shop"])  # another module's: the pool keeps both
        _wait_for(lambda: pool._ready.qsize() == 2)
        second = pool()
        assert second.given == ("shop", "tax")
        second.shutdown()


def test_an_interpreter_that_started_before_the_pool_was_asked_for_more_is_replaced():
    release = threading.Event()

    class Slow(Provided):
        def __call__(self):
            interpreter = super().__call__()
            if len(self.made) == 1:
                release.wait(30)  # still starting when the pool is asked for a library
            return interpreter

    factory = Slow()
    with dspy.InterpreterPool(factory, size=1) as pool:
        _wait_for(lambda: len(factory.made) == 1)
        pool.provide_libraries(["shop"])
        release.set()
        _wait_for(lambda: pool._ready.qsize() == 1 and len(factory.made) >= 2)
        ready = pool()
        assert ready.given == ("shop",)
        ready.shutdown()
        assert factory.made[0]._process is None  # the one that started without it


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(dspy.LocalInterpreter, id="local"),
        pytest.param(dspy.PythonInterpreter, marks=pytest.mark.deno, id="deno"),  # run by CI's Deno job
    ],
)
def test_a_flex_with_libraries_tells_its_pool(factory, tmp_path, monkeypatch):
    (tmp_path / "flexpooled.py").write_text("def double(value):\n    return value * 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    source = textwrap.dedent(
        """
        from flexpooled import double


        class M(dspy.Module):
            def forward(self, value):
                return dspy.Prediction(result=double(value))
        """
    )
    with dspy.InterpreterPool(factory, size=1) as pool:
        _wait_for(lambda: pool._ready.qsize() == 1)  # started before any module said what it needs
        flex = dspy.Flex("value: int -> result: int", libraries=["flexpooled"], module_src=source, interpreter_factory=pool)
        plain = dspy.Flex("value: int -> result: int", interpreter_factory=pool, module_src=source.replace(
            "from flexpooled import double", "double = lambda value: value * 2"))

        assert flex(value=2).result == 4  # no setup of the pool for it
        assert pool._libraries == ("flexpooled",)
        _wait_for(lambda: pool._ready.qsize() == 1)
        assert flex(value=3).result == 6  # on an interpreter the pool started with the library
        assert plain(value=4).result == 8  # which serves a module without libraries as well
