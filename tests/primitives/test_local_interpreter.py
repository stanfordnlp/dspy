import contextvars
import os
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


def test_persistent_worker_is_separate_and_captures_stdout(capsys):
    with dspy.LocalInterpreter() as interpreter:
        assert interpreter.execute("import os\nos.getpid()") != os.getpid()
        interpreter.execute("remembered = 41")
        assert interpreter.execute("remembered + 1") == 42
        assert interpreter.execute("print('guest only')") == "guest only"
        assert capsys.readouterr().out == ""


def test_os_level_output_does_not_corrupt_protocol():
    with dspy.LocalInterpreter() as interpreter:
        output = interpreter.execute(
            "import os, subprocess, sys\n"
            "os.write(1, b'guest fd stdout\\n')\n"
            "os.write(2, b'guest fd stderr\\n')\n"
            "completed = subprocess.run([sys.executable, '-c', \"import os; os.write(1, b'child stdout\\\\n')\"])"
        )
        assert output == "guest fd stdout\nguest fd stderr\nchild stdout"
        assert interpreter.execute("6 * 7") == 42


def test_python_stdout_reassignment_does_not_break_later_capture():
    with dspy.LocalInterpreter() as interpreter:
        assert interpreter.execute("import io, sys\nsys.stdout = io.StringIO()\nprint('hidden')") is None
        assert interpreter.execute("print('captured')") == "captured"


def test_background_thread_is_terminal_before_it_can_cross_executions():
    calls = []
    interpreter = dspy.LocalInterpreter(tools={"record": lambda *, value: calls.append(value)})

    with pytest.raises(CodeInterpreterError, match="left background threads running"):
        interpreter.execute(
            "import threading\n"
            "def late():\n"
            "    threading.Event().wait(60)\n"
            "    record(value='late')\n"
            "threading.Thread(target=late, daemon=True).start()"
        )

    assert calls == []
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


@pytest.mark.skipif(os.name != "posix", reason="open-inode survival control requires POSIX unlink semantics")
@pytest.mark.parametrize("shutdown", [False, True])
def test_shutdown_terminates_descendant_processes(shutdown):
    with dspy.LocalInterpreter() as interpreter:
        path = Path(interpreter.execute("import os; os.path.abspath('heartbeat')"))
        interpreter.execute(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "\"import time; f = open('heartbeat', 'w'); f.write('ready'); f.flush(); "
            "time.sleep(0.7); f.write('survived'); f.flush()\"])\n"
            "while not __import__('pathlib').Path('heartbeat').exists() "
            "or not __import__('pathlib').Path('heartbeat').read_text(): time.sleep(0.01)"
        )
        # An open inode detects survival even after the session directory is removed.
        with path.open() as heartbeat:
            if shutdown:
                interpreter.shutdown()
            time.sleep(1)
            assert heartbeat.read() == ("ready" if shutdown else "readysurvived")


def test_filesystem_environment_and_network_are_confined(tmp_path, monkeypatch):
    secret = tmp_path / "synthetic-secret"
    secret.write_text("host only")
    monkeypatch.setenv("DSPY_TEST_SECRET", "synthetic value")
    with socket.socket() as listener, dspy.LocalInterpreter() as interpreter:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        assert interpreter.execute("import os; 'DSPY_TEST_SECRET' in os.environ") is False
        interpreter.execute("from pathlib import Path; Path('private').write_text('session data')")
        assert interpreter.execute("Path('private').read_text()") == "session data"
        for operation in ["read_text()", "write_text('modified')"]:
            with pytest.raises(CodeExecutionError, match=r"FileNotFoundError|PermissionError|OSError"):
                interpreter.execute(f"Path({str(secret)!r}).{operation}")
        assert (
            interpreter.execute(
                f"import socket; s = socket.socket(); s.settimeout(1); s.connect_ex(('127.0.0.1', {port}))"
            )
            != 0
        )
    assert secret.read_text() == "host only"


def test_unavailable_runtime_is_terminal_and_never_falls_back(monkeypatch, tmp_path):
    import dspy_mxc_runtime

    def unavailable():
        raise RuntimeError("native sandbox unavailable")

    monkeypatch.setattr(dspy_mxc_runtime, "executables", unavailable)
    marker = tmp_path / "must-not-execute"
    interpreter = dspy.LocalInterpreter()
    with pytest.raises(CodeInterpreterError, match="native sandbox unavailable"):
        interpreter.execute(f"open({str(marker)!r}, 'w').close()")
    assert not marker.exists()
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


def test_tools_and_typed_submit():
    fields = [{"name": "answer", "type": "str"}, {"name": "score", "type": "int"}]
    with dspy.LocalInterpreter(tools={"add": lambda *, left, right: left + right}, output_fields=fields) as interpreter:
        assert interpreter.execute("add(left=19, right=23)") == 42
        result = interpreter.execute("SUBMIT(answer='yes', score=42)")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "yes", "score": 42}


@pytest.mark.parametrize("name", ["not-valid", "class", "SUBMIT", "__builtins__"])
def test_invalid_tool_names_fail_without_starting_or_consuming_session(name):
    with pytest.raises(ValueError, match="invalid names"):
        dspy.LocalInterpreter(tools={name: lambda: 1})

    interpreter = dspy.LocalInterpreter()
    interpreter.tools[name] = lambda: 1
    with pytest.raises(ValueError, match="invalid names"):
        interpreter.execute("1")
    assert interpreter._process is None

    interpreter.tools = {}
    assert interpreter.execute("6 * 7") == 42
    interpreter.shutdown()


@pytest.mark.parametrize(
    "output_fields",
    [
        [{"name": "class"}],
        [{"name": "not-valid"}],
        [{"name": "answer"}, {"name": "answer"}],
        [{"type": "str"}],
        [{"name": "answer", "default": object()}],
    ],
)
def test_invalid_output_fields_fail_without_consuming_session(output_fields):
    with pytest.raises(ValueError):
        dspy.LocalInterpreter(output_fields=output_fields)

    interpreter = dspy.LocalInterpreter()
    assert interpreter.execute("remembered = 41") is None
    process = interpreter._process
    interpreter.output_fields = output_fields
    with pytest.raises(ValueError):
        interpreter.execute("1")
    assert interpreter._process is process

    interpreter.output_fields = [{"name": "answer"}]
    assert interpreter.execute("remembered + 1") == 42
    assert interpreter.execute("SUBMIT(answer=42)").output == {"answer": 42}
    interpreter.shutdown()


def test_reserved_variable_names_fail_without_starting_or_consuming_session():
    interpreter = dspy.LocalInterpreter(tools={"add": lambda left, right: left + right})
    for variables in ({"class": 1}, {"SUBMIT": "shadowed"}, {"__builtins__": {}}, {"add": "shadowed"}):
        with pytest.raises(CodeInterpreterError, match="invalid names"):
            interpreter.execute("1", variables=variables)
        assert interpreter._process is None

    assert interpreter.execute("len([1])") == 1
    assert interpreter.execute("add(19, 23)") == 42
    assert interpreter.execute("SUBMIT(42)").output == {"output": 42}
    interpreter.shutdown()


def test_guest_cannot_corrupt_interpreter_owned_globals_across_executions():
    with dspy.LocalInterpreter(tools={"add": lambda left, right: left + right}) as interpreter:
        interpreter.execute("__builtins__['len'] = None\nSUBMIT = None\nadd = None")
        assert interpreter.execute("len([1])") == 1
        assert interpreter.execute("add(19, 23)") == 42
        assert interpreter.execute("SUBMIT(42)").output == {"output": 42}
        with pytest.raises(CodeExecutionError, match="_builtins"):
            interpreter.execute("SUBMIT.__self__._builtins['len'] = None")
        assert interpreter.execute("len([1])") == 1


def test_errors_are_recoverable_but_timeout_is_terminal():
    interpreter = dspy.LocalInterpreter(execution_timeout=1)
    with pytest.raises(SyntaxError):
        interpreter.execute("if")
    with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
        interpreter.execute("1 / 0")
    assert interpreter.execute("6 * 7") == 42
    with pytest.raises(CodeInterpreterError, match="exceeded execution timeout"):
        interpreter.execute("import time\ntime.sleep(10)")
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")
    interpreter.shutdown()


def test_async_host_tool():
    async def add(*, left, right):
        return left + right

    with dspy.LocalInterpreter(tools={"add": add}) as interpreter:
        assert interpreter.execute("add(left=19, right=23)") == 42


def test_timed_async_host_tool_preserves_context():
    marker = contextvars.ContextVar("marker", default="missing")
    marker.set("visible")

    async def read_marker():
        return marker.get()

    with dspy.LocalInterpreter(tools={"read_marker": read_marker}, execution_timeout=1) as interpreter:
        assert interpreter.execute("read_marker()") == "visible"


def test_host_tool_within_execution_timeout():
    def slow():
        time.sleep(0.08)
        return 42

    with dspy.LocalInterpreter(tools={"slow": slow}, execution_timeout=1) as interpreter:
        assert interpreter.execute("slow()") == 42


def test_parallel_guest_code_invokes_host_tools_concurrently_and_routes_responses():
    rendezvous = threading.Barrier(2)

    def echo(value):
        rendezvous.wait(timeout=1)
        if value == "slow":
            time.sleep(0.05)
        return value

    code = """
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(echo, value) for value in ["slow", "fast"]]
[future.result() for future in futures]
"""
    with dspy.LocalInterpreter(tools={"echo": echo}, execution_timeout=2) as interpreter:
        assert interpreter.execute(code) == ["slow", "fast"]


def test_execution_timeout_does_not_wait_for_blocked_host_tool():
    tool_started = threading.Event()
    release_tool = threading.Event()

    def blocked():
        tool_started.set()
        release_tool.wait(timeout=10)
        return 42

    interpreter = dspy.LocalInterpreter(tools={"blocked": blocked}, execution_timeout=1)
    interpreter.start()
    started = time.monotonic()
    with pytest.raises(CodeInterpreterError, match="exceeded execution timeout"):
        interpreter.execute("blocked()")
    elapsed = time.monotonic() - started
    release_tool.set()

    assert tool_started.wait(timeout=5)
    assert elapsed < 5
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


@pytest.mark.skipif(os.name != "posix", reason="SIGSTOP requires POSIX")
def test_execution_timeout_includes_a_blocked_command_write():
    interpreter = dspy.LocalInterpreter(execution_timeout=0.2)
    interpreter.start()
    # Stop the entire launcher group, including its confined worker.
    os.killpg(interpreter._process.pid, signal.SIGSTOP)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(interpreter.execute, "#" + "x" * (1024 * 1024))
        try:
            with pytest.raises(CodeInterpreterError, match="exceeded execution timeout"):
                future.result(timeout=3)
        finally:
            interpreter.shutdown()
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt permits detached descendants to survive")
@pytest.mark.parametrize("pipe", ["stdout", "stdin"])
def test_detached_pipe_holder_does_not_delay_shutdown_or_deadline(pipe):
    interpreter = dspy.LocalInterpreter(execution_timeout=0.2)
    child = None
    try:
        child = interpreter.execute(
            "import __main__, subprocess, sys\n"
            "fd = __main__._protocol_output.fileno()\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(4)'], "
            "start_new_session=True, pass_fds=(fd,))\nchild.pid"
        )
        started = time.monotonic()
        if pipe == "stdin":
            os.killpg(interpreter._process.pid, signal.SIGSTOP)
            with pytest.raises(CodeInterpreterError, match="exceeded execution timeout"):
                interpreter.execute("#" + "x" * (1024 * 1024))
        else:
            interpreter.shutdown()
        assert time.monotonic() - started < 1.5
        with pytest.raises(CodeInterpreterError, match="shut down"):
            interpreter.execute("1")
    finally:
        if child is not None:
            # This finite probe owns the detached process; the interpreter cannot
            # promise to reap arbitrary Mac descendants. Release inherited pipes.
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        interpreter.shutdown()


def test_malformed_guest_tool_request_cannot_invoke_host_tool():
    calls = []
    with dspy.LocalInterpreter(tools={"record": lambda: calls.append("called")}, execution_timeout=1) as interpreter:
        # Native Python can access the wire directly; the host cannot trust worker validation.
        with pytest.raises(CodeInterpreterError, match="invalid tool request"):
            interpreter.execute(
                "import __main__, time\n"
                "__main__.send({'type': 'tool_request', 'name': 'record', 'args': [], 'kwargs': {}})\n"
                "time.sleep(5)"
            )
        assert calls == []
        with pytest.raises(CodeInterpreterError, match="shut down"):
            interpreter.execute("1")


def test_oversized_guest_response_terminates_session():
    with dspy.LocalInterpreter(execution_timeout=5) as interpreter:
        with pytest.raises(CodeInterpreterError, match="message exceeds"):
            interpreter.execute("'x' * (16 * 1024 * 1024)")
        with pytest.raises(CodeInterpreterError, match="shut down"):
            interpreter.execute("1")


def test_guest_cannot_create_unbounded_host_tool_threads():
    with dspy.LocalInterpreter(tools={"echo": lambda value: value}, execution_timeout=5) as interpreter:
        # The budget is concurrent, not a limit on total calls in a valid program.
        assert interpreter.execute("sum(echo(i) for i in range(40))") == 780

    release = threading.Event()
    finished = threading.Event()
    lock = threading.Lock()
    completed = []

    def blocked():
        release.wait(timeout=10)
        with lock:
            completed.append(True)
            if len(completed) == 32:
                finished.set()

    try:
        with dspy.LocalInterpreter(tools={"blocked": blocked}, execution_timeout=5) as interpreter:
            with pytest.raises(CodeInterpreterError, match="32 outstanding"):
                interpreter.execute(
                    "import __main__, time\n"
                    "for i in range(33):\n"
                    "    __main__.send({'type': 'tool_request', 'id': i, 'name': 'blocked', 'args': [], 'kwargs': {}})\n"
                    "time.sleep(10)"
                )
    finally:
        release.set()
    assert finished.wait(timeout=5)
    assert len(completed) == 32


def test_host_tool_interrupt_terminates_session():
    def interrupt():
        raise KeyboardInterrupt("cancelled")

    with dspy.LocalInterpreter(tools={"interrupt": interrupt}, execution_timeout=5) as interpreter:
        with pytest.raises(KeyboardInterrupt, match="cancelled"):
            interpreter.execute("interrupt()")
        with pytest.raises(CodeInterpreterError, match="shut down"):
            interpreter.execute("1")


def test_interpreter_callbacks():
    events = []

    class RecordingCallback(BaseCallback):
        def on_interpreter_execute_start(self, call_id, instance, inputs):
            events.append(("execute_start", inputs["code"]))

        def on_interpreter_execute_end(self, call_id, outputs, exception=None):
            events.append(("execute_end", outputs))

        def on_interpreter_tool_call_start(self, call_id, instance, inputs):
            events.append(("tool_start", inputs["tool_name"]))

        def on_interpreter_tool_call_end(self, call_id, outputs, exception=None):
            events.append(("tool_end", outputs))

    with dspy.LocalInterpreter(tools={"answer": lambda: 42}, callbacks=[RecordingCallback()]) as interpreter:
        assert interpreter.execute("answer()") == 42

    assert ("execute_start", "answer()") in events
    assert ("tool_start", "answer") in events
    assert ("tool_end", 42) in events
    assert ("execute_end", 42) in events


def test_rlm_uses_local_interpreter():
    calls = []

    def add(*, left: int, right: int) -> int:
        calls.append((left, right))
        return left + right

    class Actions(dspy.Predict):
        def __init__(self, signature):
            super().__init__(signature)
            self.actions = iter(
                [
                    dspy.Prediction(reasoning="calculate", code="total = add(left=19, right=23)"),
                    dspy.Prediction(reasoning="submit", code="SUBMIT(answer=str(total))"),
                ]
            )

        def forward(self, **kwargs):
            return next(self.actions)

    rlm = dspy.RLM(
        "question: str -> answer: str",
        max_iters=2,
        tools=[add],
        interpreter_factory=dspy.LocalInterpreter,
    )
    rlm.generate_action = Actions(rlm.generate_action.signature)

    assert rlm(question="What is 19 + 23?").answer == "42"
    assert calls == [(19, 23)]
    assert dspy.LocalInterpreter.execution_instructions in rlm.generate_action.signature.instructions


@pytest.mark.asyncio
async def test_async_rlm_uses_local_interpreter_with_async_tool():
    async def add(*, left: int, right: int) -> int:
        return left + right

    class Actions(dspy.Predict):
        def __init__(self, signature):
            super().__init__(signature)
            self.actions = iter(
                [
                    dspy.Prediction(reasoning="calculate", code="total = add(left=19, right=23)"),
                    dspy.Prediction(reasoning="submit", code="SUBMIT(answer=str(total))"),
                ]
            )

        async def aforward(self, **kwargs):
            return next(self.actions)

    rlm = dspy.RLM(
        "question: str -> answer: str",
        max_iters=2,
        tools=[add],
        interpreter_factory=dspy.LocalInterpreter,
    )
    rlm.generate_action = Actions(rlm.generate_action.signature)

    assert (await rlm.acall(question="What is 19 + 23?")).answer == "42"


def test_flex_uses_local_interpreter():
    class Signature(dspy.Signature):
        value: int = dspy.InputField()
        result: int = dspy.OutputField()

    module_src = """
class AddModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Predict("value: int -> result: int")

    def forward(self, **inputs):
        predicted = self.solve(value=inputs["value"])
        return dspy.Prediction(result=predicted.result)
""".strip()

    program = dspy.Flex(Signature, interpreter_factory=dspy.LocalInterpreter)
    program._bind_code(module_src)
    with dspy.context(lm=DummyLM([{"result": 42}])):
        assert program(value=20).result == 42
