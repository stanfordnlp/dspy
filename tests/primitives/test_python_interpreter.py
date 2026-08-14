import asyncio
import dataclasses
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import types
from collections import namedtuple
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import pytest
from pydantic import BaseModel, ConfigDict

import dspy.primitives.python_interpreter as python_interpreter
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import (
    LARGE_VAR_THRESHOLD,
    PythonInterpreter,
    _deno_subprocess_env,
    _find_deno_executable,
    _make_jsonable,
    _validate_deno_version,
)


class _Hit(BaseModel):
    document_id: int
    title: str


class _TimestampedHit(_Hit):
    created_at: datetime


class _UnserializableValue:
    pass


class _UnserializableModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: _UnserializableValue


pytestmark = pytest.mark.deno


def test_execute_simple_code(pooled_interpreter):
    interpreter = pooled_interpreter
    code = "print('Hello, World!')"
    result = interpreter.execute(code)
    assert result == "Hello, World!\n", "Simple print statement should return 'Hello World!\n'"


def test_import(pooled_interpreter):
    interpreter = pooled_interpreter
    code = "import math\nresult = math.sqrt(4)\nresult"
    result = interpreter.execute(code)
    assert result == 2, "Should be able to import and use math.sqrt"


def test_user_variable_definitions(pooled_interpreter):
    interpreter = pooled_interpreter
    code = "result = number + 1\nresult"
    result = interpreter.execute(code, variables={"number": 4})
    assert result == 5, "User variable assignment should work"


def test_non_finite_float_variables(pooled_interpreter):
    """Regression test: inf/-inf/nan variables must be injected as valid Python literals.

    str(float("inf")) is the bare word "inf", which is not a valid Python name, so
    injecting it as `x = inf` previously raised NameError in the sandbox.
    """
    interpreter = pooled_interpreter
    inf_code = "result = 1 if x == float('inf') else 0\nresult"
    assert interpreter.execute(inf_code, variables={"x": float("inf")}) == 1

    neg_inf_code = "result = 1 if x == float('-inf') else 0\nresult"
    assert interpreter.execute(neg_inf_code, variables={"x": float("-inf")}) == 1

    nan_code = "import math\nresult = 1 if math.isnan(x) else 0\nresult"
    assert interpreter.execute(nan_code, variables={"x": float("nan")}) == 1


def test_rejects_python_keywords_as_variable_names(pooled_interpreter):
    """Test that Python keywords are rejected as variable names."""
    interpreter = pooled_interpreter
    # These are valid Python identifiers but reserved keywords
    # Using them as variable names would cause syntax errors
    keywords_to_test = ["for", "class", "import", "def", "return", "if", "while"]

    for keyword in keywords_to_test:
        with pytest.raises(CodeInterpreterError, match="Invalid variable name"):
            interpreter.execute("print(x)", variables={keyword: 42})


def test_failure_syntax_error(pooled_interpreter):
    interpreter = pooled_interpreter
    code = "+++"
    with pytest.raises(SyntaxError, match="Invalid Python syntax"):
        interpreter.execute(code)


def test_failure_zero_division(pooled_interpreter):
    interpreter = pooled_interpreter
    code = "1+0/0"
    with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
        interpreter.execute(code)


def test_exception_args(pooled_interpreter):
    interpreter = pooled_interpreter
    token = random.randint(1, 10**9)
    code = f"raise ValueError({token})"
    with pytest.raises(CodeExecutionError, match=rf"ValueError: \[{token}\]"):
        interpreter.execute(code)


def test_generated_exception_name_cannot_spoof_interpreter_failure(pooled_interpreter):
    interpreter = pooled_interpreter
    with pytest.raises(CodeExecutionError, match="CodeInterpreterError"):
        interpreter.execute(
            "class CodeInterpreterError(Exception):\n    pass\nraise CodeInterpreterError('generated failure')"
        )
    assert interpreter.execute("2 + 2") == 4


def test_submit_with_list(pooled_interpreter):
    """Test SUBMIT() with a list argument returns FinalOutput with dict format."""

    interpreter = pooled_interpreter
    token = random.randint(1, 10**9)
    code = f"SUBMIT(['The result is', {token}])"
    result = interpreter(code)

    assert isinstance(result, FinalOutput)
    # SUBMIT now always returns a dict with "output" key for single-output default
    assert result.output == {"output": ["The result is", token]}


def test_enable_env_vars_flag():
    os.environ["FOO_TEST_ENV"] = "test_value"

    with PythonInterpreter(enable_env_vars=None) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "", "Environment variables should be inaccessible without allow-env"

    with PythonInterpreter(enable_env_vars=["FOO_TEST_ENV"]) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "test_value", "Environment variables should be accessible with allow-env"


def test_read_file_access_control(tmp_path):
    testfile_path = tmp_path / "test_temp_file.txt"
    virtual_path = f"/sandbox/{testfile_path.name}"
    with open(testfile_path, "w") as f:
        f.write("test content")

    with PythonInterpreter(enable_read_paths=[str(testfile_path)]) as interpreter:
        code = f"with open({virtual_path!r}, 'r') as f:\n    data = f.read()\ndata"
        result = interpreter.execute(code)
        assert result == "test content", "Test file should be accessible with enable_read_paths and specified file"

    with PythonInterpreter(enable_read_paths=None) as interpreter:
        code = (
            f"try:\n"
            f"    with open({virtual_path!r}, 'r') as f:\n"
            f"        data = f.read()\n"
            f"except Exception as e:\n"
            f"    data = str(e)\n"
            f"data"
        )
        result = interpreter.execute(code)
        assert "PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower(), (
            "Test file should not be accessible without enable_read_paths"
        )


def test_enable_write_flag(tmp_path):
    testfile_path = tmp_path / "test_temp_output.txt"
    virtual_path = f"/sandbox/{testfile_path.name}"

    with PythonInterpreter(enable_write_paths=None) as interpreter:
        code = (
            f"try:\n"
            f"    with open({virtual_path!r}, 'w') as f:\n"
            f"        f.write('blocked')\n"
            f"    result = 'wrote'\n"
            f"except Exception as e:\n"
            f"    result = str(e)\n"
            f"result"
        )
        result = interpreter.execute(code)
        assert "PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower(), (
            "Test file should not be writable without enable_write_paths"
        )

    with PythonInterpreter(enable_write_paths=[str(testfile_path)]) as interpreter:
        code = f"with open({virtual_path!r}, 'w') as f:\n    f.write('allowed')\n'ok'"
        result = interpreter.execute(code)
        assert result == "ok", "Test file should be writable with enable_write_paths"
    assert testfile_path.exists()
    with open(testfile_path) as f:
        assert f.read() == "allowed", "Test file outputs should match content written during execution"

    with open(testfile_path, "w") as f:
        f.write("original_content")
    with PythonInterpreter(enable_write_paths=[str(testfile_path)], sync_files=False) as interpreter:
        code = f"with open({virtual_path!r}, 'w') as f:\n    f.write('should_not_sync')\n'done_no_sync'"
        result = interpreter.execute(code)
        assert result == "done_no_sync"
    with open(testfile_path) as f:
        assert f.read() == "original_content", "File should not be changed when sync_files is False"


def test_enable_net_flag():
    test_url = "https://example.com"

    with PythonInterpreter(enable_network_access=None) as interpreter:
        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
        with pytest.raises(CodeInterpreterError, match="PythonError"):
            interpreter.execute(code)

    with PythonInterpreter(enable_network_access=["example.com"]) as interpreter:
        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
        result = interpreter.execute(code)
        assert int(result) == 200, "Network access is permitted with enable_network_access"


def test_interpreter_security_filesystem_access(tmp_path):
    """
    Verify that the interpreter cannot read arbitrary files from the host system
    unless explicitly allowed.
    """
    # 1. Create a "secret" file on the host
    secret_file = tmp_path / "secret.txt"
    secret_content = "This is a secret content"
    secret_file.write_text(secret_content)
    secret_path_str = str(secret_file.absolute())

    # 2. Attempt to read the file WITHOUT permission
    malicious_code = f"""
import js
try:
    content = js.Deno.readTextFileSync('{secret_path_str}')
    print(content)
except Exception as e:
    print(f"Error: {{e}}")
"""

    with PythonInterpreter() as interpreter:
        output = interpreter(malicious_code)
        assert "Requires read access" in output
        assert secret_content not in output

    # 3. Attempt to read the file WITH permission
    with PythonInterpreter(enable_read_paths=[secret_path_str]) as interpreter:
        output = interpreter(malicious_code)
        assert secret_content in output


def test_default_runner_cannot_read_shared_deno_cache(monkeypatch, tmp_path):
    shared_cache = tmp_path / "deno"
    shared_cache.mkdir()
    canary = shared_cache / "secret.txt"
    canary.write_text("shared cache secret")
    monkeypatch.setenv("DENO_DIR", str(shared_cache))

    with PythonInterpreter() as interpreter:
        result = interpreter.execute(
            f"""import js
try:
    js.Deno.readTextFileSync({str(canary)!r})
    result = "disclosed"
except Exception as error:
    result = str(error)
result"""
        )

    assert "disclosed" not in result
    assert "read access" in result.lower()


def test_default_runner_starts_offline_from_warm_shared_cache(monkeypatch, tmp_path):
    shared_cache = tmp_path / "deno"
    runner = str(Path(python_interpreter.__file__).with_name("runner.js"))
    env = {**_deno_subprocess_env(), "DENO_DIR": str(shared_cache)}
    subprocess.run([_find_deno_executable(), "cache", "--no-config", "--no-lock", runner], env=env, check=True)
    monkeypatch.setenv("DENO_DIR", str(shared_cache))

    with PythonInterpreter() as interpreter:
        interpreter.deno_command.insert(interpreter.deno_command.index(os.path.realpath(runner)), "--cached-only")
        assert interpreter.execute("1 + 1") == 2


def test_tools_dict_is_copied():
    """Test that tools dict is defensively copied, not stored by reference."""
    tools = {"my_tool": lambda: "result"}
    sandbox = PythonInterpreter(tools=tools)

    # Modify the original dict after construction
    tools["new_tool"] = lambda: "new"

    # The sandbox should not see the new tool
    assert "new_tool" not in sandbox.tools


def test_serialize_tuple(pooled_interpreter):
    """Test that tuples can be serialized as variables."""
    interpreter = pooled_interpreter
    result = interpreter.execute("x", variables={"x": (1, 2, 3)})
    assert result == [1, 2, 3]  # Tuples become lists in JSON


def test_serialize_set(pooled_interpreter):
    """Test that sets can be serialized as variables."""
    interpreter = pooled_interpreter
    result = interpreter.execute("sorted(x)", variables={"x": {3, 1, 2}})
    assert result == [1, 2, 3]


def test_serialize_set_mixed_types(pooled_interpreter):
    """Test that sets with mixed types can be serialized (fallback to list)."""
    interpreter = pooled_interpreter
    # Mixed types can't be sorted, so they serialize as a list in arbitrary order
    # We verify the list contains the expected elements
    result = interpreter.execute("x", variables={"x": {1, "a"}})
    assert isinstance(result, list)
    assert set(result) == {1, "a"}


def test_serialize_pydantic_variable(pooled_interpreter):
    """Pydantic instances passed via variables= should arrive in the sandbox as dicts."""
    interpreter = pooled_interpreter
    result = interpreter.execute(
        "hit['document_id']",
        variables={"hit": _Hit(document_id=7, title="abc")},
    )
    assert result == 7


def test_serialize_pydantic_nested_in_dict(pooled_interpreter):
    """Pydantic instances nested inside list/dict variables should be coerced too."""
    interpreter = pooled_interpreter
    result = interpreter.execute(
        "(data['hit']['document_id'], data['hit']['title'])",
        variables={"data": {"hit": _Hit(document_id=11, title="nested")}},
    )
    assert result == [11, "nested"]


def test_serialize_pydantic_in_list_variable(pooled_interpreter):
    """A list variable whose elements are Pydantic instances should be coerced too."""
    interpreter = pooled_interpreter
    result = interpreter.execute(
        "sum(h['document_id'] for h in hits)",
        variables={"hits": [_Hit(document_id=1, title="a"), _Hit(document_id=2, title="b")]},
    )
    assert result == 3


def test_pydantic_json_values_are_compatible_with_large_variable_injection(monkeypatch):
    """The large-variable path should serialize Pydantic's JSON-mode values."""
    value = _TimestampedHit(
        document_id=7,
        title="dated",
        created_at=datetime(2026, 5, 14, 8, 7, 27),
    )
    expected = {
        "document_id": 7,
        "title": "dated",
        "created_at": "2026-05-14T08:07:27",
    }
    interpreter = PythonInterpreter()

    assert interpreter._to_json_compatible(value) == expected

    monkeypatch.setattr("dspy.primitives.python_interpreter.LARGE_VAR_THRESHOLD", 0)
    code = interpreter._inject_variables("hit", {"hit": value})
    assert "hit = json.loads" in code
    assert json.loads(interpreter._pending_large_vars["hit"]) == expected


def test_json_mode_values_cross_every_host_to_sandbox_path():
    """Bare datetimes/enums/sets (not wrapped in a model) serialize the way they would into a
    JSON body on all three host->sandbox paths: variable injection, large-var injection, and
    tool results — instead of raising "Unsupported value type" or str()-flattening the result."""

    class Color(Enum):
        RED = "red"

    value = {"t": datetime(2026, 1, 1), "c": Color.RED}
    interpreter = PythonInterpreter()
    assert interpreter._serialize_value(value) == "{'t': '2026-01-01T00:00:00', 'c': 'red'}"
    assert interpreter._to_json_compatible(value) == {"t": "2026-01-01T00:00:00", "c": "red"}
    jsonable = _make_jsonable({**value, "s": {3, 1}})
    assert jsonable["t"] == "2026-01-01T00:00:00" and jsonable["c"] == "red"
    assert sorted(jsonable["s"]) == [1, 3]


def test_json_mode_coercion_preserves_existing_fallbacks():
    class P(NamedTuple):
        x: int
        y: int

    assert _make_jsonable(P(1, 2)) == {"x": 1, "y": 2}  # namedtuples keep field names, not [1, 2]
    obj = object()
    assert _make_jsonable(obj) is obj  # unknown values still take the json.dumps/str fallback
    with pytest.raises(CodeInterpreterError, match="Unsupported value type"):
        PythonInterpreter()._serialize_value(object())  # injection still rejects them loudly


def test_unserializable_pydantic_variable_raises_code_interpreter_error():
    """Invalid host variables should fail before the interpreter process starts."""
    interpreter = PythonInterpreter()
    value = _UnserializableModel(value=_UnserializableValue())

    with pytest.raises(CodeInterpreterError, match="Unable to serialize _UnserializableModel as JSON") as exc_info:
        interpreter.execute("value", variables={"value": value})

    assert type(exc_info.value) is CodeInterpreterError
    assert interpreter.deno_process is None


def test_deno_command_dict_raises_type_error():
    """Test that passing a dict as deno_command raises TypeError."""
    with pytest.raises(TypeError, match="deno_command must be a list"):
        PythonInterpreter(deno_command={"invalid": "dict"})


def test_custom_deno_command_is_unchanged():
    command = ["custom-deno", "run", "custom-runner.js", "argument"]

    interpreter = PythonInterpreter(deno_command=command)

    assert interpreter.deno_command == command
    assert interpreter.deno_command is not command


def test_rejects_mounts_with_the_same_guest_basename(tmp_path):
    first = tmp_path / "first" / "shared.txt"
    second = tmp_path / "second" / "shared.txt"
    first.parent.mkdir()
    second.parent.mkdir()

    with pytest.raises(CodeInterpreterError, match="unique basenames"):
        PythonInterpreter(deno_command=["deno"], enable_read_paths=[first], enable_write_paths=[second])


def test_allows_same_canonical_file_as_read_and_write_mount(tmp_path):
    path = tmp_path / "shared.txt"

    PythonInterpreter(deno_command=["deno"], enable_read_paths=[path], enable_write_paths=[path])


def test_rejects_alias_basename_colliding_with_another_file(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second" / "shared.txt"
    alias = tmp_path / "alias" / "shared.txt"
    second.parent.mkdir()
    alias.parent.mkdir()
    alias.symlink_to(first)

    with pytest.raises(CodeInterpreterError, match="unique basenames"):
        PythonInterpreter(deno_command=["deno"], enable_read_paths=[first, alias, second])


def test_custom_deno_command_preserves_environment(monkeypatch):
    monkeypatch.setenv("DENO_NO_PACKAGE_JSON", "0")
    captured = {}
    interpreter = PythonInterpreter(deno_command=["custom-deno", "run"])

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return object()

    monkeypatch.setattr(python_interpreter.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(interpreter, "_health_check", lambda: None)

    interpreter._spawn_process()

    assert captured["command"] == ["custom-deno", "run"]
    assert captured["env"]["DENO_NO_PACKAGE_JSON"] == "0"


def test_deno_subprocess_env_disables_package_json(monkeypatch):
    monkeypatch.setenv("DENO_NO_PACKAGE_JSON", "0")
    monkeypatch.setenv("DSPY_DENO_TEST_VALUE", "preserved")

    env = _deno_subprocess_env()

    assert env["DENO_NO_PACKAGE_JSON"] == "1"
    assert env["DSPY_DENO_TEST_VALUE"] == "preserved"
    assert os.environ["DENO_NO_PACKAGE_JSON"] == "0"


def test_managed_deno_package_is_preferred(monkeypatch):
    managed_deno = types.SimpleNamespace(find_deno_bin=lambda: "/managed/bin/deno")
    monkeypatch.setitem(sys.modules, "deno", managed_deno)

    assert _find_deno_executable() == "/managed/bin/deno"


def test_missing_managed_deno_binary_falls_back_to_path(monkeypatch):
    def missing_binary():
        raise FileNotFoundError

    managed_deno = types.SimpleNamespace(find_deno_bin=missing_binary)
    monkeypatch.setitem(sys.modules, "deno", managed_deno)
    monkeypatch.setattr(shutil, "which", lambda executable: "/system/bin/deno" if executable == "deno" else None)

    assert _find_deno_executable() == "/system/bin/deno"


def test_default_command_uses_managed_deno_for_info_and_run(monkeypatch, tmp_path):
    deno_executable = str(tmp_path / "managed-deno")
    seen_operations = []
    monkeypatch.setattr(python_interpreter, "_find_deno_executable", lambda: deno_executable)
    monkeypatch.setattr(
        python_interpreter,
        "_validate_deno_version",
        lambda executable: seen_operations.append(("version", executable)),
    )
    monkeypatch.setattr(
        PythonInterpreter,
        "_get_deno_dir",
        staticmethod(lambda executable: seen_operations.append(("info", executable)) or str(tmp_path / "cache")),
    )

    interpreter = PythonInterpreter()

    assert seen_operations == [("info", deno_executable)]
    assert interpreter.deno_command[:5] == [
        deno_executable,
        "run",
        "--no-config",
        "--no-lock",
        "--node-modules-dir=false",
    ]
    runner_index = interpreter.deno_command.index(os.path.realpath(interpreter._get_runner_path()))
    assert all(interpreter.deno_command.index(flag) < runner_index for flag in interpreter.deno_command[2:5])

    monkeypatch.setattr(python_interpreter.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(interpreter, "_health_check", lambda: None)
    interpreter._spawn_process()

    assert seen_operations == [("info", deno_executable), ("version", deno_executable)]


def test_default_command_revokes_shared_cache_after_startup(monkeypatch, tmp_path):
    shared_cache = tmp_path / "deno"
    monkeypatch.setattr(PythonInterpreter, "_get_deno_dir", staticmethod(lambda executable: str(shared_cache)))
    interpreter = PythonInterpreter()

    assert str(shared_cache) in next(arg for arg in interpreter.deno_command if arg.startswith("--allow-read="))
    assert f"--dspy-deno-dir={shared_cache}" in interpreter.deno_command


def test_rejects_write_paths_overlapping_runtime_files(monkeypatch, tmp_path):
    cache = tmp_path / "deno"
    monkeypatch.setattr(PythonInterpreter, "_get_deno_dir", staticmethod(lambda executable: str(cache)))

    with pytest.raises(CodeInterpreterError, match="runtime files"):
        PythonInterpreter(enable_write_paths=[cache])


@pytest.mark.parametrize("version", [(2, 0, 0), (2, 4, 5), (2, 9, 5)])
def test_accepts_supported_deno_2_versions(monkeypatch, version):
    monkeypatch.setattr(python_interpreter, "_get_deno_version", lambda executable: version)

    _validate_deno_version("/system/bin/deno")


@pytest.mark.parametrize("version", [(1, 46, 3), (3, 0, 0)])
def test_rejects_unsupported_system_deno(monkeypatch, version):
    monkeypatch.setattr(python_interpreter, "_get_deno_version", lambda executable: version)
    version_text = "\\.".join(map(str, version))

    with pytest.raises(CodeInterpreterError, match=rf"Unsupported Deno version {version_text}"):
        _validate_deno_version("/system/bin/deno")


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (0, "not a Deno version", ""),
        (0, "", "deno 2.9.5"),
        (1, "deno 2.9.5", "version probe failed"),
    ],
)
def test_rejects_invalid_deno_version_probe(monkeypatch, returncode, stdout, stderr):
    result = types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(python_interpreter.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(CodeInterpreterError, match="Unable to determine the Deno version"):
        _validate_deno_version("/fake/bin/deno")


def test_deno_version_probe_is_bounded_and_not_cached(monkeypatch):
    results = iter(
        [
            types.SimpleNamespace(returncode=0, stdout="deno 2.9.5", stderr=""),
            types.SimpleNamespace(returncode=0, stdout="deno 2.9.4", stderr=""),
        ]
    )
    seen_timeouts = []

    def fake_run(*args, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        return next(results)

    monkeypatch.setattr(python_interpreter.subprocess, "run", fake_run)

    assert python_interpreter._get_deno_version("/fake/bin/deno") == (2, 9, 5)
    assert python_interpreter._get_deno_version("/fake/bin/deno") == (2, 9, 4)
    assert seen_timeouts == [python_interpreter.DENO_PROBE_TIMEOUT_SECONDS] * 2


def test_deno_version_probe_timeout_is_reported_as_indeterminate(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(python_interpreter.subprocess, "run", timeout)

    with pytest.raises(CodeInterpreterError, match="Unable to determine the Deno version"):
        _validate_deno_version("/hanging/bin/deno")


def test_deno_info_probe_is_bounded(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"denoDir": "/cache"}))

    monkeypatch.setattr(python_interpreter.subprocess, "run", fake_run)
    PythonInterpreter._query_deno_dir.cache_clear()

    assert PythonInterpreter._query_deno_dir("/bounded/bin/deno") == "/cache"
    assert captured["timeout"] == python_interpreter.DENO_PROBE_TIMEOUT_SECONDS


def test_explicit_deno_dir_skips_info_query(monkeypatch, tmp_path):
    deno_dir = str(tmp_path / "deno-cache")
    monkeypatch.setenv("DENO_DIR", deno_dir)
    monkeypatch.setattr(
        PythonInterpreter,
        "_query_deno_dir",
        staticmethod(lambda executable: pytest.fail(f"unexpected Deno info query for {executable}")),
    )

    assert PythonInterpreter._get_deno_dir("/managed/bin/deno") == deno_dir


def test_ignores_parent_package_json_and_node_modules(monkeypatch, tmp_path):
    """A parent Node project must not redirect runner.js's Pyodide import."""
    project_dir = tmp_path / "node-project"
    runner_dir = project_dir / "runtime"
    fake_pyodide_dir = project_dir / "node_modules" / "pyodide"
    runner_dir.mkdir(parents=True)
    fake_pyodide_dir.mkdir(parents=True)

    runner_path = runner_dir / "runner.js"
    shutil.copyfile(Path(python_interpreter.__file__).with_name("runner.js"), runner_path)
    (project_dir / "package.json").write_text(json.dumps({"dependencies": {"pyodide": "0.29.4"}}))
    (fake_pyodide_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "pyodide",
                "version": "0.29.4",
                "type": "module",
                "exports": {"./pyodide.js": "./pyodide.js"},
            }
        )
    )
    (fake_pyodide_dir / "blocked.txt").write_text("ambient package was selected")
    (fake_pyodide_dir / "pyodide.js").write_text(
        'Deno.readTextFileSync(new URL("./blocked.txt", import.meta.url));\n'
        'export default { loadPyodide() { throw new Error("DSPy loaded Pyodide from parent node_modules"); } };\n'
    )

    class ParentProjectInterpreter(PythonInterpreter):
        def _get_runner_path(self):
            return str(runner_path)

    monkeypatch.chdir(runner_dir)
    monkeypatch.delenv("DENO_NO_PACKAGE_JSON", raising=False)

    with ParentProjectInterpreter() as interpreter:
        assert interpreter.execute("6 * 7") == 42

    assert not (project_dir / "deno.lock").exists()


# =============================================================================
# Typed Tool Signature Tests
# =============================================================================


def test_tool_with_typed_signature(configure_pooled_interpreter):
    """Test that tools get proper typed signatures from inspect."""

    def my_tool(query: str, limit: int = 10) -> str:
        return f"searched '{query}' with limit {limit}"

    sandbox = configure_pooled_interpreter(tools={"my_tool": my_tool})
    # Tool should be callable with typed signature
    result = sandbox.execute('my_tool(query="test", limit=5)')
    assert result == "searched 'test' with limit 5"


def test_tool_none_type_and_nested_json_default():
    def tool(value: type(None) = None, items: list = [None]):  # noqa: B006
        return [value, items]

    with PythonInterpreter(tools={"tool": tool}) as interpreter:
        assert interpreter.execute("tool()") == [None, [None]]


@pytest.mark.parametrize(
    ("field", "code", "expected"),
    [
        ({"name": "nothing", "type": "NoneType"}, "SUBMIT(nothing=None)", {"nothing": None}),
        ({"name": "FinalOutput", "type": "str"}, "SUBMIT(FinalOutput='ok')", {"FinalOutput": "ok"}),
    ],
)
def test_submit_valid_runtime_edge_cases(field, code, expected):
    with PythonInterpreter(output_fields=[field]) as interpreter:
        assert interpreter.execute(code) == FinalOutput(expected)


def test_tool_positional_args(configure_pooled_interpreter):
    """Test that tools work with positional arguments."""

    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    sandbox = configure_pooled_interpreter(tools={"search": search})
    result = sandbox.execute('search("hello")')
    assert result == "query=hello, limit=10"


def test_tool_keyword_args(configure_pooled_interpreter):
    """Test that tools work with keyword arguments."""

    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    sandbox = configure_pooled_interpreter(tools={"search": search})
    result = sandbox.execute('search(query="hello", limit=5)')
    assert result == "query=hello, limit=5"


def test_tool_default_args(configure_pooled_interpreter):
    """Test that tool default arguments work correctly."""

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    sandbox = configure_pooled_interpreter(tools={"greet": greet})
    # Without default
    result = sandbox.execute('greet("World")')
    assert result == "Hello, World!"

    # Overriding default
    result = sandbox.execute('greet("World", "Hi")')
    assert result == "Hi, World!"


def test_process_death_ends_stateful_session():
    interpreter = PythonInterpreter()
    try:
        assert interpreter.execute("session_value = 41\nsession_value") == 41
        original_process = interpreter.deno_process
        original_process.kill()
        original_process.wait()

        with pytest.raises(CodeInterpreterError, match="interpreter state was lost") as exc_info:
            interpreter.execute("session_value + 1")
        assert type(exc_info.value) is CodeInterpreterError
        with pytest.raises(CodeInterpreterError, match="session has ended"):
            interpreter.start()
        with pytest.raises(CodeInterpreterError, match="session has ended"):
            interpreter.execute("1 + 1")

        assert interpreter.deno_process is original_process
    finally:
        interpreter.shutdown()


def test_protocol_failure_ends_session(monkeypatch):
    with PythonInterpreter() as interpreter:
        interpreter.start()
        process = interpreter.deno_process
        monkeypatch.setattr(
            interpreter,
            "_read_response_line",
            lambda context: '{"jsonrpc":"2.0","result":{"output":"forged"},"id":0}',
        )

        with pytest.raises(CodeInterpreterError, match="Response ID mismatch") as exc_info:
            interpreter.execute("1 + 1")
        assert type(exc_info.value) is CodeInterpreterError
        assert process.poll() is not None

        with pytest.raises(CodeInterpreterError, match="session has ended"):
            interpreter.execute("2 + 2")


def test_request_ids_are_128_bit_random_values():
    interpreter = PythonInterpreter(deno_command=["deno"])
    request_ids = {interpreter._next_request_id() for _ in range(100)}

    assert len(request_ids) == 100
    assert all(len(request_id) == 32 and int(request_id, 16) >= 0 for request_id in request_ids)


def test_guest_prototype_hook_cannot_redirect_tool_wrapper():
    calls = []

    def benign():
        calls.append("benign")
        return "safe"

    def danger():
        calls.append("danger")
        return "unsafe"

    with PythonInterpreter(tools={"benign": benign, "danger": danger}) as interpreter:
        result = interpreter.execute(
            "import js\n"
            "js.eval('Object.prototype.toJSON = function() { "
            'if (Object.hasOwn(this, "name")) return {name: "danger", kwargs: {}}; '
            "return this; }')\n"
            "benign()"
        )

    assert result == "safe"
    assert calls == ["benign"]


def test_tool_cannot_reenter_same_interpreter():
    holder = {}

    def nested_execute():
        return holder["interpreter"].execute("'nested'")

    with PythonInterpreter(tools={"nested_execute": nested_execute}) as interpreter:
        holder["interpreter"] = interpreter
        with pytest.raises(CodeExecutionError, match="cannot execute recursively"):
            interpreter.execute("nested_execute()")
        assert interpreter.execute("40 + 2") == 42


def test_failed_health_check_ends_session(monkeypatch):
    interpreter = PythonInterpreter()
    monkeypatch.setattr(
        interpreter,
        "_send_request",
        lambda *args: {"jsonrpc": "2.0", "result": {"output": "unexpected"}, "id": 1},
    )

    try:
        with pytest.raises(CodeInterpreterError, match="Unexpected ping response"):
            interpreter.start()
        assert interpreter.deno_process.poll() is not None

        with pytest.raises(CodeInterpreterError, match="session has ended"):
            interpreter.start()
    finally:
        interpreter.shutdown()


def test_shutdown_ends_session():
    interpreter = PythonInterpreter()
    interpreter.start()
    interpreter.shutdown()

    with pytest.raises(CodeInterpreterError, match="session has ended") as exc_info:
        interpreter.start()
    assert type(exc_info.value) is CodeInterpreterError
    with pytest.raises(CodeInterpreterError, match="session has ended"):
        interpreter.execute("1 + 1")


def test_tool_all_positional_args(configure_pooled_interpreter):
    """Test that tools work when all arguments are passed positionally."""

    def add(a: int, b: int, c: int) -> str:
        return f"{a + b + c}"

    sandbox = configure_pooled_interpreter(tools={"add": add})
    result = sandbox.execute("add(1, 2, 3)")
    assert result == "6"

    # Mixed: some positional, some keyword
    result = sandbox.execute("add(10, 20, c=30)")
    assert result == "60"


def test_tool_error_surfaces_as_runtime_error(configure_pooled_interpreter):
    """Test that exceptions raised by a tool surface as RuntimeError in the sandbox."""

    def failing_tool(x: int) -> str:
        raise ValueError(f"bad value: {x}")

    sandbox = configure_pooled_interpreter(tools={"failing_tool": failing_tool})
    result = sandbox.execute(
        "try:\n"
        "    failing_tool(42)\n"
        "    output = 'no error'\n"
        "except RuntimeError as e:\n"
        "    output = str(e)\n"
        "output"
    )
    assert "ValueError" in result
    assert "bad value: 42" in result


def test_tool_async_def_function(configure_pooled_interpreter):
    """async def tools should be awaited so the sandbox sees the resolved value."""

    async def slow_search(query: str) -> str:
        await asyncio.sleep(0)
        return f"answer:{query}"

    sandbox = configure_pooled_interpreter(tools={"slow_search": slow_search})
    result = sandbox.execute("slow_search(query='hello')")
    assert result == "answer:hello"


def test_tool_async_def_raises_propagates(configure_pooled_interpreter):
    """Exceptions raised inside an async tool should surface as RuntimeError in the sandbox."""

    async def failing_async(x: int) -> str:
        await asyncio.sleep(0)
        raise ValueError(f"boom:{x}")

    sandbox = configure_pooled_interpreter(tools={"failing_async": failing_async})
    result = sandbox.execute(
        "try:\n"
        "    failing_async(7)\n"
        "    output = 'no error'\n"
        "except RuntimeError as e:\n"
        "    output = str(e)\n"
        "output"
    )
    assert "ValueError" in result
    assert "boom:7" in result


# =============================================================================
# Tool Return Type Tests
# =============================================================================


def test_tool_returning_int(configure_pooled_interpreter):
    """Test that tools returning int preserve the type in the sandbox."""

    def count_items(label: str) -> int:
        return 4

    sandbox = configure_pooled_interpreter(tools={"count_items": count_items})
    result = sandbox.execute(
        'n = count_items(label="pages")\n'
        'print(type(n).__name__)\n'
        'print(n + 1)'
    )
    assert "int" in result
    assert "5" in result


def test_tool_returning_float(configure_pooled_interpreter):
    """Test that tools returning float preserve the type in the sandbox."""

    def get_score() -> float:
        return 0.95

    sandbox = configure_pooled_interpreter(tools={"get_score": get_score})
    result = sandbox.execute(
        "x = get_score()\n"
        "print(type(x).__name__)\n"
        "print(x * 2)"
    )
    assert "float" in result
    assert "1.9" in result


def test_tool_returning_bool(configure_pooled_interpreter):
    """Test that tools returning bool preserve the type in the sandbox."""

    def is_valid() -> bool:
        return True

    sandbox = configure_pooled_interpreter(tools={"is_valid": is_valid})
    result = sandbox.execute(
        'v = is_valid()\n'
        'print(type(v).__name__)\n'
        'print(v and "yes")'
    )
    assert "bool" in result
    assert "yes" in result


def test_tool_returning_none(configure_pooled_interpreter):
    """Test that tools returning None yield an empty string in the sandbox.

    Pyodide does not map JS null to Python None (it becomes JsNull), so
    None results are sent as empty strings to match existing behavior.
    """

    def do_nothing() -> None:
        return None

    sandbox = configure_pooled_interpreter(tools={"do_nothing": do_nothing})
    result = sandbox.execute(
        "v = do_nothing()\n"
        "print(type(v).__name__)\n"
        "print(repr(v))"
    )
    assert "str" in result
    assert "''" in result


def test_tool_returning_list(configure_pooled_interpreter):
    """Test that tools returning list preserve the type in the sandbox."""

    def get_pages() -> list:
        return [1, 2, 3]

    sandbox = configure_pooled_interpreter(tools={"get_pages": get_pages})
    result = sandbox.execute(
        "pages = get_pages()\n"
        "print(type(pages).__name__)\n"
        "print(pages[0] + 10)"
    )
    assert "list" in result
    assert "11" in result


def test_tool_returning_dict(configure_pooled_interpreter):
    """Test that tools returning dict preserve the type in the sandbox."""

    def get_info() -> dict:
        return {"count": 4, "label": "pages"}

    sandbox = configure_pooled_interpreter(tools={"get_info": get_info})
    result = sandbox.execute(
        'info = get_info()\n'
        'print(type(info).__name__)\n'
        'print(info["count"] + 1)'
    )
    assert "dict" in result
    assert "5" in result


def test_tool_returning_non_json_serializable(configure_pooled_interpreter):
    """Test that tools returning non-JSON-serializable objects fall back to string."""

    class Custom:
        def __str__(self):
            return "custom-object"

    def get_custom() -> object:
        return Custom()

    sandbox = configure_pooled_interpreter(tools={"get_custom": get_custom})
    result = sandbox.execute(
        "v = get_custom()\n"
        "print(v)"
    )
    assert "custom-object" in result


def test_tool_returning_nan_falls_back_to_string(configure_pooled_interpreter):
    """Test that tools returning float('nan') or float('inf') fall back to string.

    These values are not valid JSON, so they should go through the str()
    fallback path rather than breaking JSON.parse in the sandbox.
    """

    def get_nan() -> float:
        return float("nan")

    def get_inf() -> float:
        return float("inf")

    sandbox = configure_pooled_interpreter(tools={"get_nan": get_nan, "get_inf": get_inf})
    result = sandbox.execute(
        "n = get_nan()\n"
        "print(type(n).__name__)\n"
        "print(n)"
    )
    assert "str" in result
    assert "nan" in result

    result = sandbox.execute(
        "i = get_inf()\n"
        "print(type(i).__name__)\n"
        "print(i)"
    )
    assert "str" in result
    assert "inf" in result


def test_tool_returns_pydantic_model(configure_pooled_interpreter):
    """Pydantic models returned from a tool should arrive in the sandbox as dicts."""

    def search() -> _TimestampedHit:
        return _TimestampedHit(
            document_id=42,
            title="example",
            created_at=datetime(2026, 5, 14, 8, 7, 27),
        )

    sandbox = configure_pooled_interpreter(tools={"search": search})
    result = sandbox.execute("r = search()\n(r['document_id'], r['title'], r['created_at'])")
    assert result == [42, "example", "2026-05-14T08:07:27"]


def test_tool_returns_list_of_pydantic_models(configure_pooled_interpreter):
    """Lists of Pydantic models from a tool should round-trip as lists of dicts."""

    def search_many() -> list[_Hit]:
        return [_Hit(document_id=1, title="a"), _Hit(document_id=2, title="b")]

    sandbox = configure_pooled_interpreter(tools={"search_many": search_many})
    result = sandbox.execute(
        "hits = search_many()\nsum(h['document_id'] for h in hits)"
    )
    assert result == 3


def test_tool_returning_unserializable_pydantic_model_raises_execution_error(configure_pooled_interpreter):
    """A BaseModel that cannot honor JSON transport should remain a visible tool error."""

    def get_value() -> _UnserializableModel:
        return _UnserializableModel(value=_UnserializableValue())

    sandbox = configure_pooled_interpreter(tools={"get_value": get_value})
    with pytest.raises(
        CodeExecutionError,
        match="CodeInterpreterError.*Unable to serialize _UnserializableModel as JSON",
    ):
        sandbox.execute("get_value()")


# -- dataclass tool returns --------------------------------------------------


@dataclasses.dataclass
class _BBox:
    x0: float
    top: float
    x1: float
    bottom: float


@dataclasses.dataclass
class _PageTable:
    index: int
    bbox: _BBox
    strategy: str


def test_tool_returns_dataclass(configure_pooled_interpreter):
    """Dataclass instances returned from a tool should arrive as dicts."""

    def get_bbox() -> _BBox:
        return _BBox(x0=18.2, top=108.0, x1=554.4, bottom=748.8)

    sandbox = configure_pooled_interpreter(tools={"get_bbox": get_bbox})
    result = sandbox.execute("b = get_bbox()\n(b['x0'], b['bottom'])")
    assert result == [18.2, 748.8]


def test_tool_returns_nested_dataclass(configure_pooled_interpreter):
    """Nested dataclass instances should be recursively converted to dicts."""

    def get_table() -> _PageTable:
        return _PageTable(index=0, bbox=_BBox(x0=1.0, top=2.0, x1=3.0, bottom=4.0), strategy="lines")

    sandbox = configure_pooled_interpreter(tools={"get_table": get_table})
    result = sandbox.execute("t = get_table()\n(t['bbox']['x0'], t['strategy'])")
    assert result == [1.0, "lines"]


def test_tool_returns_list_of_dataclasses(configure_pooled_interpreter):
    """Lists of dataclass instances should round-trip as lists of dicts."""

    def get_tables() -> list[_PageTable]:
        return [
            _PageTable(index=0, bbox=_BBox(x0=1.0, top=2.0, x1=3.0, bottom=4.0), strategy="lines"),
            _PageTable(index=1, bbox=_BBox(x0=5.0, top=6.0, x1=7.0, bottom=8.0), strategy="text"),
        ]

    sandbox = configure_pooled_interpreter(tools={"get_tables": get_tables})
    result = sandbox.execute("ts = get_tables()\n[t['index'] for t in ts]")
    assert result == [0, 1]


# -- namedtuple tool returns -------------------------------------------------


class _TypedPoint(NamedTuple):
    x: float
    y: float
    label: str


_Point = namedtuple("_Point", ["x", "y"])


def test_tool_returns_typing_namedtuple(configure_pooled_interpreter):
    """typing.NamedTuple instances should arrive as dicts."""

    def get_point() -> _TypedPoint:
        return _TypedPoint(x=10.5, y=20.3, label="origin")

    sandbox = configure_pooled_interpreter(tools={"get_point": get_point})
    result = sandbox.execute("p = get_point()\n(p['x'], p['label'])")
    assert result == [10.5, "origin"]


def test_tool_returns_collections_namedtuple(configure_pooled_interpreter):
    """collections.namedtuple instances should arrive as dicts."""

    def get_point() -> _Point:
        return _Point(x=3.0, y=4.0)

    sandbox = configure_pooled_interpreter(tools={"get_point": get_point})
    result = sandbox.execute("p = get_point()\np['x'] + p['y']")
    assert result == 7.0


def test_tool_returns_dataclass_with_unserializable_field_falls_back(configure_pooled_interpreter):
    """Dataclass with non-serializable fields should fall back to str() gracefully."""

    @dataclasses.dataclass
    class _Holder:
        name: str
        lock: threading.Lock

    def get_holder() -> _Holder:
        return _Holder(name="test", lock=threading.Lock())

    sandbox = configure_pooled_interpreter(tools={"get_holder": get_holder})
    result = sandbox.execute("h = get_holder()\ntype(h).__name__")
    assert result == "str"


# =============================================================================
# Multi-Output SUBMIT Tests
# =============================================================================


def test_submit_with_typed_signature(configure_pooled_interpreter):
    """Test SUBMIT with typed output signature."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]

    sandbox = configure_pooled_interpreter(output_fields=output_fields)
    result = sandbox.execute('SUBMIT(answer="the answer", confidence=0.95)')

    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_positional_args(configure_pooled_interpreter):
    """Test SUBMIT with positional arguments."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]

    sandbox = configure_pooled_interpreter(output_fields=output_fields)
    result = sandbox.execute('SUBMIT("the answer", 0.95)')

    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_multi_output(configure_pooled_interpreter):
    """Test SUBMIT with multiple output fields using positional args."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]

    sandbox = configure_pooled_interpreter(output_fields=output_fields)
    # Positional args: values mapped to output fields in order
    code = """
a = "my answer"
s = 42
SUBMIT(a, s)
"""
    result = sandbox.execute(code)

    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "my answer", "score": 42}


def test_submit_wrong_arg_count(configure_pooled_interpreter):
    """Test SUBMIT with wrong number of args gives clear error."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]

    sandbox = configure_pooled_interpreter(output_fields=output_fields)
    with pytest.raises(CodeInterpreterError) as exc_info:
        sandbox.execute("x = 1; SUBMIT(x)")  # Only 1 arg, expects 2
    assert "missing 1 required positional argument" in str(exc_info.value)


def test_extract_parameters():
    """Test that _extract_parameters correctly extracts function signatures."""

    def example_fn(required: str, optional: int = 5, untyped=None) -> str:
        pass

    sandbox = PythonInterpreter()
    params = sandbox._extract_parameters(example_fn)

    assert len(params) == 3
    assert params[0] == {"name": "required", "type": "str"}
    assert params[1] == {"name": "optional", "type": "int", "default": 5}
    assert params[2] == {"name": "untyped", "default": None}


def test_extract_parameters_complex_types():
    """Test that _extract_parameters handles complex types gracefully."""

    def complex_fn(items: list | None = None, data: dict[str, int] | None = None) -> list:
        pass

    sandbox = PythonInterpreter()
    params = sandbox._extract_parameters(complex_fn)

    assert len(params) == 2
    # Complex types like Union are not included in type annotation
    assert params[0] == {"name": "items", "default": None}
    assert params[1] == {"name": "data", "default": None}


# =============================================================================
# Large Variable Injection Tests
# =============================================================================


def test_large_variable_injection(pooled_interpreter):
    """Test that large strings are injected via filesystem to avoid Pyodide's FFI size limit."""
    # Create a string just over the threshold
    large_data = "x" * (LARGE_VAR_THRESHOLD + 1024)

    interpreter = pooled_interpreter
    result = interpreter.execute("len(data)", variables={"data": large_data})
    assert result == len(large_data), "Large variable should be correctly injected and accessible"


def test_large_variable_content_integrity(pooled_interpreter):
    """Test that large variable content is preserved exactly through filesystem injection."""
    # Create a string with recognizable pattern just over threshold
    pattern = "ABCDEFGHIJ" * 100
    large_data = pattern * ((LARGE_VAR_THRESHOLD // len(pattern)) + 1)

    interpreter = pooled_interpreter
    # Check first and last parts to verify content integrity
    code = """
first_100 = data[:100]
last_100 = data[-100:]
(first_100, last_100)
"""
    result = interpreter.execute(code, variables={"data": large_data})
    assert result[0] == large_data[:100], "First 100 chars should match"
    assert result[1] == large_data[-100:], "Last 100 chars should match"


def test_mixed_small_and_large_variables(pooled_interpreter):
    """Test that small and large variables can be used together."""
    small_var = "hello"
    large_var = "x" * (LARGE_VAR_THRESHOLD + 1024)

    interpreter = pooled_interpreter
    code = "f'{small} has {len(small)} chars, large has {len(large)} chars'"
    result = interpreter.execute(code, variables={"small": small_var, "large": large_var})
    expected = f"{small_var} has {len(small_var)} chars, large has {len(large_var)} chars"
    assert result == expected, "Both small and large variables should work together"


def test_multiple_large_variables(pooled_interpreter):
    """Test that multiple large variables can be injected."""
    large_a = "a" * (LARGE_VAR_THRESHOLD + 100)
    large_b = "b" * (LARGE_VAR_THRESHOLD + 200)

    interpreter = pooled_interpreter
    code = "(len(var_a), len(var_b), var_a[0], var_b[0])"
    result = interpreter.execute(code, variables={"var_a": large_a, "var_b": large_b})
    assert result == [len(large_a), len(large_b), "a", "b"], "Multiple large variables should work"


def test_large_list_variable(pooled_interpreter):
    """Test that large list variables are injected via filesystem and JSON parsed."""
    # Each element "x" serializes to ~3 chars, so divide threshold by 3
    num_elements = LARGE_VAR_THRESHOLD // 3
    large_list = ["x"] * num_elements

    interpreter = pooled_interpreter
    code = "(len(data), data[0], data[-1], type(data).__name__)"
    result = interpreter.execute(code, variables={"data": large_list})
    assert result == [num_elements, "x", "x", "list"]


def test_nested_sets_and_tuples(pooled_interpreter):
    """Test that nested structures with sets and tuples are converted to JSON-compatible types."""
    complex_data = {"tags": {1, 2, 3}, "coords": (10, 20), "nested": [{"inner_set": {"a", "b"}}]}

    interpreter = pooled_interpreter
    result = interpreter.execute("data", variables={"data": complex_data})
    # Sets become sorted lists, tuples become lists
    assert result["tags"] == [1, 2, 3]
    assert result["coords"] == [10, 20]
    assert result["nested"][0]["inner_set"] == ["a", "b"]


def test_small_variable_not_using_filesystem():
    """Test that small variables are embedded in code, not using filesystem."""
    small_var = "small string"

    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}  # Initialize
    interpreter._inject_variables("print(x)", {"x": small_var})

    assert interpreter._pending_large_vars == {}, "Small variables should not be in _pending_large_vars"


def test_large_variable_threshold_boundary():
    """Test behavior at exactly the threshold boundary.

    The threshold applies to the serialized size, not the original value.
    For strings, serialization adds 2 bytes (quotes).
    """
    # Serialized size at threshold - should use embedded (not filesystem)
    # Account for 2 bytes of quotes added by repr()
    at_threshold = "x" * (LARGE_VAR_THRESHOLD - 2)

    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": at_threshold})
    assert interpreter._pending_large_vars == {}, "Serialized size at threshold should be embedded"

    # Serialized size over threshold - should use filesystem
    over_threshold = "x" * (LARGE_VAR_THRESHOLD - 1)
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": over_threshold})
    assert "x" in interpreter._pending_large_vars, "Serialized size over threshold should use filesystem"


def test_enable_read_paths_symlink(tmp_path):
    """Regression test for #9501: symlinked enable_read_paths must resolve so Deno
    can read through them (denoland/deno#9607 — Deno prefix-matches against the
    realpath of the file being read). The sandbox virtual path keeps the user's
    original basename so user code refers to the file by the name passed in.
    """
    real_file = tmp_path / "real_name.txt"
    real_file.write_text("through symlink")
    link_file = tmp_path / "link_name.txt"
    try:
        link_file.symlink_to(real_file)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with PythonInterpreter(enable_read_paths=[str(link_file)]) as interp:
        allow_read_arg = next(a for a in interp.deno_command if a.startswith("--allow-read="))
        allow_read = allow_read_arg[len("--allow-read="):].split(",")
        assert os.path.realpath(str(real_file)) in allow_read
        assert str(link_file) not in allow_read

        result = interp.execute("with open('/sandbox/link_name.txt') as f:\n    data = f.read()\ndata")
        assert result == "through symlink"


def test_enable_read_paths_multiple_files(tmp_path):
    """Test that enable_read_paths works with multiple files in the same directory.

    Regression test for bug where mounting multiple files to /sandbox/ failed
    because Pyodide's ErrnoError has errno but no message property, causing
    the 'directory exists' check to fail on the second file.
    """
    file1 = tmp_path / "test1.txt"
    file2 = tmp_path / "test2.txt"
    file3 = tmp_path / "test3.txt"
    file1.write_text("Content 1")
    file2.write_text("Content 2")
    file3.write_text("Content 3")

    with PythonInterpreter(enable_read_paths=[str(file1), str(file2), str(file3)]) as interpreter:
        code = (
            "import os\n"
            "files = sorted(os.listdir('/sandbox'))\n"
            "contents = {}\n"
            "for f in files:\n"
            "    with open(f'/sandbox/{f}') as fh:\n"
            "        contents[f] = fh.read()\n"
            "(files, contents)"
        )
        result = interpreter.execute(code)
        files, contents = result

        assert files == ["test1.txt", "test2.txt", "test3.txt"], "All three files should be mounted"
        assert contents["test1.txt"] == "Content 1"
        assert contents["test2.txt"] == "Content 2"
        assert contents["test3.txt"] == "Content 3"


def test_system_exit_is_recoverable_and_session_stays_synced(pooled_interpreter):
    """Regression test for #10165: the unhandled Deno rejection accompanying
    SystemExit must not be consumed as the response, desyncing later requests."""
    interpreter = pooled_interpreter
    with pytest.raises(CodeExecutionError, match="SystemExit"):
        interpreter.execute("import sys\nsys.exit(0)")

    assert interpreter.execute("print('still alive')") == "still alive\n"


def test_keyboard_interrupt_is_recoverable_and_session_stays_synced(pooled_interpreter):
    """KeyboardInterrupt takes the same asyncio re-raise path as SystemExit (#10165)."""
    interpreter = pooled_interpreter
    with pytest.raises(CodeExecutionError, match="KeyboardInterrupt"):
        interpreter.execute("raise KeyboardInterrupt('stop')")

    assert interpreter.execute("print('still alive')") == "still alive\n"


def test_base_exception_subclasses_are_recoverable_and_session_stays_synced(pooled_interpreter):
    """Subclasses of SystemExit/KeyboardInterrupt take the same re-raise path (#10165)."""
    interpreter = pooled_interpreter
    for code, error_type in [
        ("class ExitSignal(SystemExit): pass\nraise ExitSignal(0)", "ExitSignal"),
        ("class InterruptSignal(KeyboardInterrupt): pass\nraise InterruptSignal()", "InterruptSignal"),
    ]:
        with pytest.raises(CodeExecutionError, match=error_type):
            interpreter.execute(code)
        assert interpreter.execute("print('still alive')") == "still alive\n"


def test_out_of_band_messages_are_skipped_not_consumed_as_responses():
    """Notifications and unsolicited id-less errors are diagnostics, not responses (#10165)."""
    interpreter = PythonInterpreter()

    notification = {"jsonrpc": "2.0", "method": "unhandled_error", "params": {"message": "boom"}}
    assert interpreter._handle_out_of_band_message(notification, "during test")
    assert interpreter._last_diagnostic == "boom"

    unsolicited_error = {"jsonrpc": "2.0", "error": {"code": -32007, "message": "crash"}, "id": None}
    assert interpreter._handle_out_of_band_message(unsolicited_error, "during test")
    assert interpreter._last_diagnostic == "crash"

    # Real responses (success or error, with an id) must not be consumed.
    result = {"jsonrpc": "2.0", "result": {"output": "2\n"}, "id": 1}
    assert not interpreter._handle_out_of_band_message(result, "during test")
    error_response = {"jsonrpc": "2.0", "error": {"code": -32007, "message": "boom"}, "id": 1}
    assert not interpreter._handle_out_of_band_message(error_response, "during test")


def test_id_less_protocol_errors_are_terminal():
    """ParseError/InvalidRequest mean the sandbox never read the request, so no
    response will follow; waiting for one would block forever (#10165)."""
    interpreter = PythonInterpreter()
    parse_error = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Invalid JSON input"}, "id": None}
    with pytest.raises(CodeInterpreterError, match="Protocol error"):
        interpreter._handle_out_of_band_message(parse_error, "during test")


def test_unsolicited_error_line_is_not_consumed_as_the_response(monkeypatch):
    """#10165: an id-less async error arriving ahead of the real response (the
    wire trace an unhandled rejection produces) must not be mistaken for it."""
    request_id = "a" * 32
    monkeypatch.setattr(python_interpreter.secrets, "token_hex", lambda _: request_id)

    class FakeDeno:
        def __init__(self, lines):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("".join(line + "\n" for line in lines))
            self.stderr = io.StringIO()

        def poll(self):
            return None

    interpreter = PythonInterpreter()
    interpreter.deno_process = FakeDeno([
        json.dumps({"jsonrpc": "2.0", "error": {"code": -32007, "message": "Unhandled async error: PythonError"}, "id": None}),
        json.dumps({"jsonrpc": "2.0", "result": {"output": "ok\n"}, "id": request_id}),
    ])
    assert interpreter.execute("print('ok')") == "ok\n"


def test_base_exceptions_do_not_desync_interpreter():
    with PythonInterpreter() as interpreter:
        for code, error_type in [
            ("import sys\nsys.exit(0)", "SystemExit"),
            ("raise KeyboardInterrupt()", "KeyboardInterrupt"),
            ("class ExitSignal(SystemExit): pass\nraise ExitSignal(0)", "ExitSignal"),
            ("class InterruptSignal(KeyboardInterrupt): pass\nraise InterruptSignal()", "InterruptSignal"),
        ]:
            with pytest.raises(CodeExecutionError, match=error_type):
                interpreter.execute(code)
            assert interpreter.execute("print('still alive')") == "still alive\n"


def test_execution_instructions_are_class_metadata():
    interpreter = PythonInterpreter(deno_command=["deno", "run"])

    assert interpreter.execution_instructions == PythonInterpreter.execution_instructions
    assert "Pyodide" in interpreter.execution_instructions
