import io
import json

from dspy.primitives.python_interpreter import PythonInterpreter


class FakeProcess:
    def __init__(self, stdout_lines):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(stdout_lines))
        self.stderr = io.StringIO()

    def poll(self):
        return None


def test_send_request_skips_unsolicited_parse_error():
    interpreter = PythonInterpreter(deno_command=["deno", "run", "runner.js"])
    interpreter.deno_process = FakeProcess(
        [
            json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Invalid JSON input"}, "id": None})
            + "\n",
            json.dumps({"jsonrpc": "2.0", "result": {"output": "2\n"}, "id": 1}) + "\n",
        ]
    )

    response = interpreter._send_request("execute", {"code": "print(1+1)"}, "during health check")

    assert response == {"jsonrpc": "2.0", "result": {"output": "2\n"}, "id": 1}
