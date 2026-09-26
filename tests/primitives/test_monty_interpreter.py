import pytest

import dspy

pytest.importorskip("pydantic_monty")


@pytest.fixture
def interpreter():
    interpreter = dspy.MontyInterpreter(output_fields=[{"name": "answer"}])
    yield interpreter
    interpreter.shutdown()


def test_state_output_and_recovery(interpreter):
    assert interpreter.execute("values = [3, 8]\ndef total(): return sum(values)") is None
    with pytest.raises(dspy.CodeExecutionError, match="ZeroDivisionError"):
        interpreter.execute("1 / 0")
    assert interpreter.execute("print('total:', total())") == "total: 11"
    with pytest.raises(SyntaxError):
        interpreter.execute("if :")
    assert interpreter.execute("total()") == 11


def test_submit_stops_and_allows_another_iteration(interpreter):
    calls = []
    interpreter.tools["record"] = lambda: calls.append(True)
    result = interpreter.execute("value = 13\nSUBMIT(answer=value)\nrecord()")
    assert result == dspy.FinalOutput({"answer": 13})
    assert calls == []
    assert interpreter.execute("SUBMIT(answer=value + 6)") == dspy.FinalOutput({"answer": 19})
    with pytest.raises(dspy.CodeExecutionError, match="fields"):
        interpreter.execute("SUBMIT(wrong=0)")
    assert interpreter.execute("value") == 13


def test_tools_can_change_and_recover_from_errors(interpreter):
    interpreter.tools["tool"] = lambda x, scale=3: x * scale
    assert interpreter.execute("tool(7, scale=4)") == 28
    interpreter.tools["tool"] = lambda x, scale=3: x + scale
    assert interpreter.execute("tool(7)") == 10
    interpreter.tools["tool"] = lambda: 1 / 0
    assert interpreter.execute("try:\n    tool()\nexcept Exception:\n    print('recovered')") == "recovered"


@pytest.mark.asyncio
async def test_async_host_tool_in_running_loop(interpreter):
    async def tool(value: int) -> int:
        return value * 3

    interpreter.tools["tool"] = tool
    assert interpreter.execute("tool(7)") == 21


def test_shutdown_is_terminal(interpreter):
    interpreter.start()
    interpreter.start()
    interpreter.shutdown()
    interpreter.shutdown()
    with pytest.raises(dspy.CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


def test_configured_factory_is_lazy_and_creates_independent_sessions():
    constructed = []

    class TrackedMonty(dspy.MontyInterpreter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            constructed.append(self)

    factory = TrackedMonty.configured(output_fields=[{"name": "answer"}])
    assert constructed == []
    assert factory.flex_execution_instructions == TrackedMonty.flex_execution_instructions
    with dspy.context(interpreter_factory=factory):
        first = dspy.primitives.code_interpreter._create_interpreter(None)
        second = dspy.primitives.code_interpreter._create_interpreter(None)
    try:
        assert len(constructed) == 2 and first is not second
        first.execute("value = 13")
        with pytest.raises(dspy.CodeExecutionError, match="NameError"):
            second.execute("value")
        first.shutdown()
        assert second.execute("SUBMIT(answer=29)") == dspy.FinalOutput({"answer": 29})
    finally:
        first.shutdown()
        second.shutdown()


def test_configured_rejects_unknown_constructor_options():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        dspy.MontyInterpreter.configured(working_directory="/work")


@pytest.mark.parametrize("limits,code", [
    ({"max_feed_duration_secs": 0.01}, "while True: pass"),
    ({"max_memory": 100_000}, "large = 'x' * 1_000_000"),
])
def test_resource_limits_end_the_session(limits, code):
    interpreter = dspy.MontyInterpreter(limits=limits)
    try:
        with pytest.raises(dspy.CodeInterpreterError, match="resource limit") as error:
            interpreter.execute(code)
        assert not isinstance(error.value, dspy.CodeExecutionError)
        with pytest.raises(dspy.CodeInterpreterError, match="shut down"):
            interpreter.execute("1")
    finally:
        interpreter.shutdown()


def test_tool_positional_varargs_and_kwargs(interpreter):
    def tool(first, *rest, scale=1, **extras):
        return (first + sum(rest) + sum(extras.values())) * scale

    interpreter.tools["tool"] = tool
    assert interpreter.execute("tool(2, 7, 11, scale=3, extra=5)") == 75


def test_host_interrupt_discards_suspended_session(interpreter):
    def interrupt():
        raise KeyboardInterrupt

    interpreter.tools["interrupt"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        interpreter.execute("interrupt()")
    with pytest.raises(dspy.CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


def test_virtual_files_and_cwd_persist_across_feeds():
    from pathlib import PurePosixPath

    from pydantic_monty import MemoryFile, OSAccess

    fs = OSAccess([MemoryFile("/work/input.txt", "13"), MemoryFile("/work/sub/input.txt", "29")])
    interpreter = dspy.MontyInterpreter(os=fs, cwd="/work")
    try:
        interpreter.start()  # Explicit startup must not lose the initial cwd.
        assert interpreter.execute("open('input.txt').read()") == "13"
        interpreter.execute("import os\nos.chdir('sub')\nfrom pathlib import Path\nPath('result.txt').write_text('41')")
        assert interpreter.execute("[os.getcwd(), open('input.txt').read(), Path('result.txt').read_text()]") == [
            "/work/sub", "29", "41",
        ]
    finally:
        interpreter.shutdown()
    assert fs.path_read_text(PurePosixPath("/work/sub/result.txt")) == "41"


def test_filesystem_is_denied_without_capabilities(interpreter, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("host only")
    with pytest.raises(dspy.CodeExecutionError, match="PermissionError"):
        interpreter.execute(f"open({str(secret)!r}).read()")


@pytest.mark.parametrize("mode", ["read-only", "read-write", "overlay"])
def test_mount_modes_and_caller_ownership(tmp_path, mode):
    from pydantic_monty import MountDir

    source = tmp_path / "input.txt"
    source.write_text("original")
    with MountDir(host_path=tmp_path, virtual_path="/data", mode=mode) as mount:
        # A session shutdown must not close the mount needed by the next session.
        for _ in range(2):
            interpreter = dspy.MontyInterpreter(mount=mount, cwd="/data")
            try:
                assert interpreter.execute("open('input.txt').read()") == "original"
                code = "from pathlib import Path\nPath('output.txt').write_text('edited')\nPath('output.txt').read_text()"
                if mode == "read-only":
                    with pytest.raises(dspy.CodeExecutionError, match="PermissionError"):
                        interpreter.execute(code)
                else:
                    assert interpreter.execute(code) == "edited"
                # Overlay writes must not be mistaken for persistent RLM scratch space.
                assert interpreter.execute("from pathlib import Path\nPath('output.txt').exists()") == (
                    mode == "read-write"
                )
            finally:
                interpreter.shutdown()
    assert (tmp_path / "output.txt").exists() == (mode == "read-write")
    assert source.read_text() == "original"


def test_mount_confines_traversal_and_symlinks(tmp_path):
    from pydantic_monty import MountDir

    root = tmp_path / "allowed"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("outside mount")
    (root / "link.txt").symlink_to("../secret.txt")
    with MountDir(host_path=root, virtual_path="/data", mode="read-only") as mount:
        interpreter = dspy.MontyInterpreter(mount=mount, cwd="/data")
        try:
            for path in ("../secret.txt", "link.txt"):
                with pytest.raises(dspy.CodeExecutionError, match="PermissionError"):
                    interpreter.execute(f"open({path!r}).read()")
        finally:
            interpreter.shutdown()
