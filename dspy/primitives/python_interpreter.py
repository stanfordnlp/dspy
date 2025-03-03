import json
import subprocess
from typing import Any, Dict, List, Optional
import os

class InterpreterError(ValueError):
    pass

class PythonInterpreter:
    r"""
    PythonInterpreter that runs code in a sandboxed environment using Deno and Pyodide.

    Prerequisites:
    - Deno (https://docs.deno.com/runtime/getting_started/installation/).

    Example Usage:
    ```python
    code_string = "print('Hello'); 1 + 2"
    interp = PythonInterpreter()
    output = interp(code_string)
    print(output)  # If final statement is non-None, prints the numeric result, else prints captured output
    interp.shutdown()
    ```
    """

    def __init__(
        self,
        deno_command: Optional[List[str]] = None
    ) -> None:
        if isinstance(deno_command, dict):
            deno_command = None  # no-op, just a guard in case someone passes a dict
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
                    "> curl -fsSL https://deno.land/install.sh | sh\n"
                    "*or*, on macOS with Homebrew:\n"
                    "> brew install deno\n"
                    "For additional configurations: https://docs.deno.com/runtime/getting_started/installation/"
                )
                raise InterpreterError(install_instructions) from e

    def _inject_variables(self, code: str, variables: Dict[str, Any]) -> str:
        # Insert Python assignments for each variable at the top of the code
        injected_lines = []
        for key, value in variables.items():
            if not key.isidentifier():
                raise InterpreterError(f"Invalid variable name: '{key}'")
            python_value = self._serialize_value(value)
            injected_lines.append(f"{key} = {python_value}")
        injected_code = "\n".join(injected_lines) + "\n" + code
        return injected_code

    def _serialize_value(self, value: Any) -> str:
        # Basic safe serialization
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

        # Send the code as JSON
        input_data = json.dumps({"code": code})
        try:
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
        except BrokenPipeError:
            # If the process died, restart and try again once
            self._ensure_deno_process()
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()

        # Read one JSON line from stdout
        output_line = self.deno_process.stdout.readline().strip()
        if not output_line:
            # Possibly the subprocess died or gave no output
            err_output = self.deno_process.stderr.read()
            raise InterpreterError(f"No output from Deno subprocess. Stderr: {err_output}")

        # Parse that line as JSON
        try:
            result = json.loads(output_line)
        except json.JSONDecodeError:
            # If not valid JSON, just return raw text
            result = {"output": output_line}

        # If we have an error, determine if it's a SyntaxError or other error using error.errorType.
        if "error" in result:
            error_msg = result["error"]
            error_type = result.get("errorType", "Sandbox Error")
            if error_type == "SyntaxError":
                raise SyntaxError(f"Invalid Python syntax. message: {error_msg}")
            else:
                raise InterpreterError(f"{error_type}: {error_msg}")

        # If there's no error, return the "output" field
        return result.get("output", None)

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
