import subprocess

import pytest

from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, _bind_interpreter
from dspy.primitives.python_interpreter import PythonInterpreter
from tests.mock_interpreter import MockInterpreter


def test_execution_instructions_are_read_only():
    interpreter = PythonInterpreter(deno_command=["deno", "run"])

    assert isinstance(interpreter.execution_instructions, str)
    assert interpreter.execution_instructions
    with pytest.raises(AttributeError):
        interpreter.execution_instructions = "replacement"


def test_construct_and_read_execution_instructions_does_not_invoke_deno(monkeypatch):
    PythonInterpreter._get_deno_dir.cache_clear()

    def fail(*args, **kwargs):
        raise AssertionError("constructing the interpreter or reading instructions invoked Deno")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)

    interpreter = PythonInterpreter()

    assert interpreter.execution_instructions == interpreter.execution_instructions
    assert interpreter.deno_process is None


def test_execution_instructions_describe_stable_constraints_without_host_paths(tmp_path):
    secret_path = tmp_path / "distinctive-host-path"
    interpreter = PythonInterpreter(enable_read_paths=[secret_path])
    instructions = interpreter.execution_instructions

    assert "Pyodide" in instructions
    assert "WebAssembly" in instructions
    assert "persist" in instructions
    assert "subprocesses are unavailable" in instructions
    assert "network access are unavailable unless explicitly enabled" in instructions
    assert str(secret_path) not in instructions


def test_legacy_interpreter_still_satisfies_runtime_protocol():
    assert isinstance(MockInterpreter(), CodeInterpreter)


def test_bind_copies_and_replaces_tools_and_output_fields():
    def first():
        return "first"

    tools = {"first": first}
    output_fields = [{"name": "answer", "type": "str"}]
    interpreter = PythonInterpreter(deno_command=["deno", "run"])

    interpreter.bind(tools=tools, output_fields=output_fields)
    tools["second"] = lambda: "second"
    output_fields[0]["name"] = "changed"

    assert interpreter.tools == {"first": first}
    assert interpreter.output_fields == [{"name": "answer", "type": "str"}]

    def second():
        return "second"

    interpreter.bind(tools={"second": second})

    assert interpreter.tools == {"second": second}
    assert interpreter.output_fields is None


@pytest.mark.parametrize("name", ["not valid", "class", "SUBMIT"])
def test_bind_rejects_invalid_tool_names_without_changing_binding(name):
    def original():
        return "original"

    interpreter = PythonInterpreter(deno_command=["deno", "run"], tools={"original": original})

    with pytest.raises(CodeInterpreterError):
        interpreter.bind(tools={name: lambda: None})

    assert interpreter.tools == {"original": original}


def test_bind_rejects_invalid_output_fields_without_changing_binding():
    original_fields = [{"name": "answer", "type": "str"}]
    interpreter = PythonInterpreter(deno_command=["deno", "run"], output_fields=original_fields)

    with pytest.raises(CodeInterpreterError, match="Duplicate output field"):
        interpreter.bind(tools={}, output_fields=[{"name": "answer"}, {"name": "answer"}])

    assert interpreter.output_fields == original_fields


def test_bind_after_shutdown_fails():
    interpreter = PythonInterpreter(deno_command=["deno", "run"])
    interpreter.shutdown()

    with pytest.raises(CodeInterpreterError, match="session has ended"):
        interpreter.bind(tools={})


def test_bind_registration_explicitly_restores_default_submit(monkeypatch):
    registrations = []
    interpreter = PythonInterpreter(deno_command=["deno", "run"])
    monkeypatch.setattr(
        interpreter,
        "_send_request",
        lambda method, params, context: registrations.append((method, params, context)),
    )

    interpreter.bind(tools={}, output_fields=[{"name": "answer", "type": "str"}])
    interpreter._register_tools()
    interpreter.bind(tools={}, output_fields=None)
    interpreter._register_tools()

    assert registrations[0][1]["outputs"] == [{"name": "answer", "type": "str"}]
    assert registrations[1][1]["outputs"] == []


def test_bind_helper_uses_compatible_hook():
    class BindAwareInterpreter(MockInterpreter):
        def __init__(self):
            super().__init__()
            self.bind_calls = []

        def bind(self, *, tools, output_fields=None):
            self.bind_calls.append((tools, output_fields))

    interpreter = BindAwareInterpreter()
    tools = {"tool": lambda: "ok"}
    output_fields = [{"name": "answer"}]

    _bind_interpreter(interpreter, tools=tools, output_fields=output_fields)

    assert interpreter.bind_calls == [(tools, output_fields)]


def test_bind_helper_ignores_unrelated_bind_method():
    class LegacyInterpreter(MockInterpreter):
        def bind(self, address):
            raise AssertionError("unrelated bind method should not be called")

    interpreter = LegacyInterpreter(tools={"old": lambda: "old"})

    def new_tool():
        return "new"

    _bind_interpreter(interpreter, tools={"new": new_tool})

    assert interpreter.tools == {"new": new_tool}
