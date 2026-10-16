"""Opt-in integration checks; require real installed runtimes, not sandbox mocks."""

import base64
import concurrent.futures
import os
import socket
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from cmpnd import CmpndCallback
from PIL import Image

import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError
from dspy.utils.dummies import DummyLM
from scripts.mxc_interpreter.interpreter import MXCInterpreter


@pytest.fixture
def factory():
    runtime = os.environ.get("MXC_EXECUTABLE")
    if not runtime:
        pytest.fail("Set MXC_EXECUTABLE to the installed runtime; see README.md")
    return MXCInterpreter.configured(runtime=runtime)


@pytest.fixture(autouse=True)
def tracing():
    with dspy.context(callbacks=[CmpndCallback(capture_inputs=False, capture_outputs=False)]):
        yield


def test_missing_dependency_never_runs_generated_code(factory, tmp_path):
    marker = tmp_path / "must-not-run"
    worker = factory(bwrap=str(tmp_path / "missing-bwrap"))
    try:
        with pytest.raises(CodeInterpreterError, match="Bubblewrap executable not found"):
            worker.execute(f"from pathlib import Path; Path({str(marker)!r}).touch()")
        assert not marker.exists()
        with pytest.raises(CodeInterpreterError, match="shut down"):
            worker.execute("1")
    finally:
        worker.shutdown()


def test_python_native_libraries_and_persistence(factory):
    with factory() as worker:
        assert worker.execute("""
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
from PIL import Image
@dataclass
class Box:
    values: list
    def __getitem__(self, key): return self.values[key]
    def __setitem__(self, key, value): self.values[key] = value
class Derived(Box):
    @property
    def total(self): return sum(self.values)
@contextmanager
def box():
    yield Derived([2, 11, 7])
with box() as state:
    state[1] += 9
    product = int(np.array(state.values) @ np.array([3, 5, 2]))
    image = Image.new('RGB', (3, 2), 'red').transpose(Image.Transpose.ROTATE_90)
[state.total, product, list(image.size)]
""") == [29, 120, [2, 3]]
        assert worker.execute("state[1]") == 20
        with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
            worker.execute("1/0")
        assert worker.execute("state.total") == 29


def test_parallel_tools_typed_submit_and_files(factory):
    barrier = threading.Barrier(2)

    def echo(value):
        barrier.wait(timeout=2)
        if value == "slow":
            time.sleep(0.05)
        return value

    async def add(left, right):
        return left + right

    with factory(tools={"echo": echo, "add": add}, output_fields=[{"name": "answer", "type": "int"}]) as worker:
        assert worker.execute("""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(echo, value) for value in ['slow', 'fast']]
    values = [f.result() for f in futures]
Path('answer.txt').write_text(str(add(17, 26)))
values
""") == ["slow", "fast"]
        assert worker.execute("SUBMIT(answer=int(Path('answer.txt').read_text()))").output == {"answer": 43}


def test_flex_module_predictor_and_native_packages(factory):
    program = dspy.Flex("value: int -> result: int", interpreter_factory=factory)
    program._bind_code("""
class Calculate(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Predict('value: int -> result: int')
    def forward(self, **inputs):
        import numpy as np
        from PIL import Image
        class Scale:
            def __getitem__(self, key): return key * 3
        with Image.new('RGB', (3, 2), 'red') as image:
            value = int(np.array([inputs['value'], Scale()[4]]) @ np.array([2, 5])) + image.width
        result = self.solve(value=value)
        return dspy.Prediction(result=result['result'] + 1)
""")
    lm = DummyLM([{"result": 104}])
    with dspy.context(lm=lm):
        assert program(value=20).result == 105
    assert "103" in str(lm.history[-1]["messages"])


def test_recursive_rlm(factory):
    rlm = dspy.RLM("query -> answer", max_iters=1, interpreter_factory=factory)
    rlm.generate_action = lambda **kw: dspy.Prediction(
        reasoning="nested",
        code="res = dspy.RLM('question -> answer', max_iters=1)(question='hi')\nSUBMIT(res.answer)",
    )
    with dspy.context(lm=DummyLM([{"plan": "return", "code": "SUBMIT('nested worked')"}])):
        assert rlm(query="q").answer == "nested worked"


def test_rlm_subagent_and_host_tool(factory):
    calls = []

    def echo(text: str) -> str:
        """Echo a string."""
        calls.append(text)
        return text

    rlm = dspy.RLM("query -> answer", tools=[echo], max_iters=1, interpreter_factory=factory)
    rlm.generate_action = lambda **kw: dspy.Prediction(
        reasoning="delegate",
        code="res = dspy.ReActV2('question -> answer', tools=[echo])(question='hi')\nSUBMIT(res.answer)",
    )
    lm = DummyLM(
        [
            {
                "next_thought": "echo",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "echo", "args": {"text": "hi"}},
                    ]
                ),
            },
            {
                "next_thought": "submit",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "submit", "args": {"answer": "echoed hi"}},
                    ]
                ),
            },
        ]
    )
    with dspy.context(lm=lm):
        assert rlm(query="q").answer == "echoed hi"
    assert calls == ["hi"]


def test_native_image_roundtrip(factory):
    image = Image.new("RGB", (3, 2), "red")
    image.putpixel((0, 0), (0, 255, 0))
    source = BytesIO()
    image.save(source, format="PNG")
    with factory() as worker:
        encoded = worker.execute(
            """
import base64, cv2, numpy as np
image = cv2.imdecode(np.frombuffer(base64.b64decode(source), dtype=np.uint8), cv2.IMREAD_COLOR)
edited = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
ok, png = cv2.imencode('.png', edited)
assert ok
base64.b64encode(png).decode()
""",
            variables={"source": base64.b64encode(source.getvalue()).decode()},
        )
    edited = Image.open(BytesIO(base64.b64decode(encoded)))
    assert edited.size == (2, 3)
    assert edited.getpixel((1, 0)) == (0, 255, 0)
    assert edited.getpixel((0, 2)) == (255, 0, 0)


@pytest.mark.asyncio
async def test_async_rlm(factory):
    async def add(left: int, right: int) -> int:
        return left + right

    rlm = dspy.RLM("query -> answer", tools=[add], max_iters=1, interpreter_factory=factory)

    async def action(**kw):
        return dspy.Prediction(reasoning="add", code="SUBMIT(str(add(left=17, right=26)))")

    rlm.generate_action = type("Actions", (), {"acall": staticmethod(action)})()
    assert (await rlm.acall(query="sum")).answer == "43"


def test_secret_file_environment_and_native_io_denied(factory, tmp_path, monkeypatch):
    secret = tmp_path / "synthetic-secret"
    secret.write_text("canary-not-a-real-secret")
    monkeypatch.setenv("DSPY_SYNTHETIC_SECRET", "canary-not-a-real-secret")
    with factory() as worker:
        assert worker.execute("import os; 'DSPY_SYNTHETIC_SECRET' in os.environ") is False
        # MXC otherwise exposes the host's entire /etc as a read-only baseline.
        assert worker.execute("from pathlib import Path; list(Path('/etc').iterdir())") == []
        worker.execute(f"from pathlib import Path; secret = {str(secret)!r}")
        for code in [
            "Path(secret).read_text()",
            "Path(secret).write_text('modified')",
            "Path('escape').symlink_to(secret); Path('escape').read_text()",
        ]:
            with pytest.raises(CodeExecutionError, match=r"FileNotFoundError|PermissionError|OSError"):
                worker.execute(code)
        assert worker.execute("import ctypes; ctypes.CDLL(None).open(secret.encode(), 0)") == -1
        # Zero-byte append even on an unexpected success cannot damage the runtime.
        bootstrap = worker.session / "worker.py"
        with bootstrap.open("ab"):
            pass  # Positive control: the host can write this file.
        with pytest.raises(CodeExecutionError, match=r"PermissionError|OSError"):
            worker.execute(f"open({str(bootstrap)!r}, 'ab').close()")
    assert secret.read_text() == "canary-not-a-real-secret"


def test_host_network_denied(factory):
    with socket.socket() as listener, factory() as worker:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass  # Positive control: this listener really is reachable on the host.
        assert (
            worker.execute(f"import socket; s = socket.socket(); s.settimeout(1); s.connect_ex(('127.0.0.1', {port}))")
            != 0
        )
        with pytest.raises(CodeExecutionError):
            worker.execute("import urllib.request; urllib.request.urlopen('http://example.invalid', timeout=2)")


def test_concurrent_sessions_cannot_read_each_other(factory):
    with factory() as first, factory() as second:
        first.execute("from pathlib import Path; Path('private').write_text('first only')")
        second.execute("from pathlib import Path; Path('private').write_text('second only')")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda worker: worker.execute("Path('private').read_text()"), [first, second]))
        assert values == ["first only", "second only"]
        with pytest.raises(CodeExecutionError, match=r"FileNotFoundError|PermissionError"):
            second.execute(f"Path({str(first.workspace / 'private')!r}).read_text()")


@pytest.mark.parametrize("stop", ["none", "shutdown", "timeout"])
def test_detached_descendant_lifetime(factory, stop):
    with factory(execution_timeout=1) as worker:
        worker.execute("""
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c',
    "import time; f = open('marker', 'w'); f.write('ready'); f.flush(); "
    "time.sleep(1.5); f.write('survived'); f.flush()"], start_new_session=True)
while not Path('marker').exists() or not Path('marker').read_text(): time.sleep(0.01)
""")
        # Keep the inode open: removal of the workspace must not hide a surviving child.
        with (worker.workspace / "marker").open() as marker:
            if stop == "shutdown":
                worker.shutdown()
            elif stop == "timeout":
                with pytest.raises(CodeInterpreterError, match="timeout"):
                    worker.execute("while True: pass")
            time.sleep(1.8)
            assert marker.read() == ("readysurvived" if stop == "none" else "ready")


def test_shared_temp_is_hidden(factory):
    shared = Path("/tmp/claude")
    created = not shared.exists()
    shared.mkdir(exist_ok=True)
    try:
        with factory() as worker:
            assert worker.execute("from pathlib import Path; Path('/tmp/claude').exists()") is False
    finally:
        if created:
            shared.rmdir()
