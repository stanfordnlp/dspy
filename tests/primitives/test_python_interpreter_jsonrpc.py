import io

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.python_interpreter import PythonInterpreter


class FakeProcess:
    def __init__(self, stdout_line: str):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout_line + "\n")
        self.stderr = io.StringIO()

    def poll(self):
        return None


def test_send_request_reports_null_id_jsonrpc_error_message():
    interpreter = PythonInterpreter(deno_command=["deno", "run", "runner.js"])
    interpreter.deno_process = FakeProcess(
        '{"jsonrpc":"2.0","error":{"code":-32700,"message":"Invalid JSON input"},"id":null}'
    )

    with pytest.raises(CodeInterpreterError, match="Error during health check: Invalid JSON input"):
        interpreter._send_request("execute", {"code": "print(1+1)"}, "during health check")


def test_send_request_still_rejects_mismatched_error_ids():
    interpreter = PythonInterpreter(deno_command=["deno", "run", "runner.js"])
    interpreter.deno_process = FakeProcess(
        '{"jsonrpc":"2.0","error":{"code":-32099,"message":"wrong response"},"id":999}'
    )

    with pytest.raises(CodeInterpreterError, match="Response ID mismatch during health check: expected 1, got 999"):
        interpreter._send_request("execute", {"code": "print(1+1)"}, "during health check")
