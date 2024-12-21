import json
import subprocess
from typing import Any, Dict, List, Optional
import os

class InterpreterError(ValueError):
    pass

class PythonInterpreter:
    r"""
    PythonInterpreter that runs code in a sandboxed environment using Deno and Pyodide.
    Adapted from "Simon Willison’s TILs" (https://til.simonwillison.net/deno/pyodide-sandbox)

    Prerequisites:
    - Deno (https://docs.deno.com/runtime/getting_started/installation/).

    Example Usage: 
    ```python
    code_string = "4 + 5"
    output =PythonInterpreter()(code_string)
    print(output)
    ```
    """

    def __init__(
        self, 
        deno_command: Optional[List[str]] = None
    ) -> None:
        if isinstance(deno_command, dict):
            deno_command = None #no-op
        self.deno_command = deno_command or [
            "deno", "run", "--allow-read", self._get_runner_path()
        ]
        self.deno_process = None

    def _get_runner_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "runner.js")

    def _ensure_deno_process(self) -> None:
        if self.deno_process is None or self.deno_process.poll() is not None:
            try:
                self.deno_process = subprocess.Popen(
                    self.deno_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except FileNotFoundError as e:
                install_instructions = (
                    "Deno executable not found. Please install Deno to proceed.\n"
                    "Installation instructions:\n"
                    "curl -fsSL https://deno.land/install.sh | sh\n"
                    "For additional configurations: https://docs.deno.com/runtime/getting_started/installation/"
                )
                raise InterpreterError(install_instructions) from e

    def _inject_variables(self, code: str, variables: Dict[str, Any]) -> str:
        injected_lines = []
        for key, value in variables.items():
            if not key.isidentifier():
                raise InterpreterError(f"Invalid variable name: '{key}'")
            python_value = self._serialize_value(value)
            injected_lines.append(f"{key} = {python_value}")
        injected_code = "\n".join(injected_lines) + "\n" + code
        return injected_code

    def _serialize_value(self, value: Any) -> str:
        if isinstance(value, str):
            return repr(value)
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif value is None:
            return 'None'
        elif isinstance(value, list) or isinstance(value, dict):
            return json.dumps(value)
        else:
            raise InterpreterError(f"Unsupported value type: {type(value).__name__}")

    def execute(
        self, 
        code: str, 
        variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        variables = variables or {}
        code = self._inject_variables(code, variables)
        self._ensure_deno_process()
        input_data = json.dumps({"code": code})
        try:
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
        except BrokenPipeError:
            self._ensure_deno_process()
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
        output_line = self.deno_process.stdout.readline().strip()
        if not output_line:
            err_output = self.deno_process.stderr.read()
            raise InterpreterError(f"No output from Deno subprocess. Stderr: {err_output}")
        try:
            result = json.loads(output_line)
        except json.JSONDecodeError:
            result = {"output": output_line}
        if not isinstance(result, dict):
            result = {"output": result}
        if 'error' in result:
            raise InterpreterError(f"Sandbox Error: {result['error']}")
        return result.get('output', None)

    def __call__(
        self, 
        code: str, 
        variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.execute(code, variables)

    def shutdown(self) -> None:
        if self.deno_process and self.deno_process.poll() is None:
            shutdown_message = json.dumps({"shutdown": True}) + "\n"
            self.deno_process.stdin.write(shutdown_message)
            self.deno_process.stdin.flush()
            self.deno_process.stdin.close()
            self.deno_process.wait()
            self.deno_process = None
