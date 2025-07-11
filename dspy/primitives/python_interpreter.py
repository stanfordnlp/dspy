import json
import os
import subprocess
from os import PathLike
from types import TracebackType
from typing import Any


class InterpreterError(RuntimeError):
    pass


class PythonInterpreter:
    r"""
    PythonInterpreter that runs code in a sandboxed environment using Deno and Pyodide.

    Prerequisites:
    - Deno (https://docs.deno.com/runtime/getting_started/installation/).

    Example Usage:
    ```python
    code_string = "print('Hello'); 1 + 2"
    with PythonInterpreter() as interp:
        output = interp(code_string) # If final statement is non-None, prints the numeric result, else prints captured output
    ```
    """

    def __init__(
        self,
        deno_command: list[str] | None = None,
        enable_read_paths: list[PathLike | str] | None = None,
        enable_write_paths: list[PathLike | str] | None = None,
        enable_env_vars: list[str] | None = None,
        enable_network_access: list[str] | None = None,
        sync_files: bool = True,
    ) -> None:
        """
        Args:
            deno_command: command list to launch Deno.
            enable_read_paths: Files or directories to allow reading from in the sandbox.
            enable_write_paths: Files or directories to allow writing to in the sandbox.
            enable_env_vars: Environment variable names to allow in the sandbox.
            enable_network_access: Domains or IPs to allow network access in the sandbox.
            sync_files: If set, syncs changes within the sandbox back to original files after execution.
        """
        if isinstance(deno_command, dict):
            deno_command = None  # no-op, just a guard in case someone passes a dict

        self.enable_read_paths = enable_read_paths or []
        self.enable_write_paths = enable_write_paths or []
        self.enable_env_vars = enable_env_vars or []
        self.enable_network_access = enable_network_access or []
        self.sync_files = sync_files
        # TODO later on add enable_run (--allow-run) by proxying subprocess.run through Deno.run() to fix 'emscripten does not support processes' error

        if deno_command:
            self.deno_command = list(deno_command)
        else:
            args = ["deno", "run", "--allow-read"]
            self._env_arg  = ""
            if self.enable_env_vars:
                user_vars = [str(v).strip() for v in self.enable_env_vars]
                args.append("--allow-env=" + ",".join(user_vars))
                self._env_arg = ",".join(user_vars)
            if self.enable_network_access:
                args.append(f"--allow-net={','.join(str(x) for x in self.enable_network_access)}")
            if self.enable_write_paths:
                args.append(f"--allow-write={','.join(str(x) for x in self.enable_write_paths)}")

            args.append(self._get_runner_path())

            # For runner.js to load in env vars
            if self._env_arg:
                args.append(self._env_arg)
            self.deno_command = args

        self.deno_process = None
        self._mounted_files = False

    def _get_runner_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "runner.js")

    def _mount_files(self):
        if self._mounted_files:
            return
        paths_to_mount = []
        if self.enable_read_paths:
            paths_to_mount.extend(self.enable_read_paths)
        if self.enable_write_paths:
            paths_to_mount.extend(self.enable_write_paths)
        if not paths_to_mount:
            return
        for path in paths_to_mount:
            if not path:
                continue
            if not os.path.exists(path):
                if self.enable_write_paths and path in self.enable_write_paths:
                    open(path, "a").close()
                else:
                    raise FileNotFoundError(f"Cannot mount non-existent file: {path}")
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            mount_msg = json.dumps({"mount_file": str(path), "virtual_path": virtual_path})
            self.deno_process.stdin.write(mount_msg + "\n")
            self.deno_process.stdin.flush()
        self._mounted_files = True

    def _sync_files(self):
        if not self.enable_write_paths or not self.sync_files:
            return
        for path in self.enable_write_paths:
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            sync_msg = json.dumps({
                "sync_file": virtual_path,
                "host_file": str(path)
            })
            self.deno_process.stdin.write(sync_msg + "\n")
            self.deno_process.stdin.flush()


    def _ensure_deno_process(self) -> None:
        if self.deno_process is None or self.deno_process.poll() is not None:
            try:
                self.deno_process = subprocess.Popen(
                    self.deno_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="UTF-8",
                    env=os.environ.copy()
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

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
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
            return "None"
        elif isinstance(value, list) or isinstance(value, dict):
            return json.dumps(value)
        else:
            raise InterpreterError(f"Unsupported value type: {type(value).__name__}")

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        variables = variables or {}
        code = self._inject_variables(code, variables)
        self._ensure_deno_process()
        self._mount_files()

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
            if error_type == "FinalAnswer":
                # The `FinalAnswer` trick to receive output from the sandbox interpreter,
                # just simply replace the output with the arguments.
                result["output"] = result.get("errorArgs", None)
            elif error_type == "SyntaxError":
                raise SyntaxError(f"Invalid Python syntax. message: {error_msg}")
            else:
                raise InterpreterError(f"{error_type}: {result.get('errorArgs') or error_msg}")

        # If there's no error or got `FinalAnswer`, return the "output" field
        self._sync_files()
        return result.get("output", None)

    def __enter__(self):
        return self

    # All exception fields are ignored and the runtime will automatically re-raise the exception
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ):
        self.shutdown()

    def __call__(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
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
