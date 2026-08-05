import subprocess

import pytest

from dspy.primitives.code_interpreter import CodeInterpreter
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
