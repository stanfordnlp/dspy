"""
Local interpreter for secure Python code execution using Deno/Pyodide.

This module provides PythonInterpreter, which runs Python code in a sandboxed
WASM environment using Deno and Pyodide. It implements the Interpreter
protocol defined in interpreter.py.
"""

import functools
import inspect
import json
import keyword
import logging
import os
import subprocess
from os import PathLike
from typing import Any, Callable

from dspy.primitives.code_interpreter import SIMPLE_TYPES, CodeInterpreterError, FinalOutput

__all__ = ["PythonInterpreter", "FinalOutput", "CodeInterpreterError"]

logger = logging.getLogger(__name__)


class PythonInterpreter:
    """Local interpreter for secure Python execution using Deno and Pyodide.

    Implements the Interpreter protocol for secure code execution in a
    WASM-based sandbox. Code runs in an isolated Pyodide environment with
    no access to the host filesystem, network, or environment by default.

    Prerequisites:
        Deno must be installed: https://docs.deno.com/runtime/getting_started/installation/

    Example:
        ```python
        # Basic execution
        with PythonInterpreter() as interp:
            result = interp("print(1 + 2)")  # Returns "3"

        # With host-side tools
        def my_tool(question: str) -> str:
            return "answer"

        with PythonInterpreter(tools={"my_tool": my_tool}) as interp:
            result = interp("print(my_tool(question='test'))")
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
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
    ) -> None:
        """
        Args:
            deno_command: command list to launch Deno.
            enable_read_paths: Files or directories to allow reading from in the sandbox.
            enable_write_paths: Files or directories to allow writing to in the sandbox.
                All write paths will also be able to be read from for mounting.
            enable_env_vars: Environment variable names to allow in the sandbox.
            enable_network_access: Domains or IPs to allow network access in the sandbox.
            sync_files: If set, syncs changes within the sandbox back to original files after execution.
            tools: Dictionary mapping tool names to callable functions.
                   Each function should accept keyword arguments and return a string.
                   Tools are callable directly from sandbox code by name.
            output_fields: List of output field definitions for typed SUBMIT signature.
                   Each dict should have 'name' and optionally 'type' keys.
        """
        if isinstance(deno_command, dict):
            raise TypeError("deno_command must be a list of strings, not a dict")

        self.enable_read_paths = enable_read_paths or []
        self.enable_write_paths = enable_write_paths or []
        self.enable_env_vars = enable_env_vars or []
        self.enable_network_access = enable_network_access or []
        self.sync_files = sync_files
        self.tools = dict(tools) if tools else {}
        self.output_fields = output_fields
        self._tools_registered = False
        # TODO later on add enable_run (--allow-run) by proxying subprocess.run through Deno.run() to fix 'emscripten does not support processes' error

        if deno_command:
            self.deno_command = list(deno_command)
        else:
            args = ["deno", "run"]

            # Allow reading runner.js and explicitly enabled paths
            allowed_read_paths = [self._get_runner_path()]

            # Also allow reading Deno's cache directory so Pyodide can load its files
            deno_dir = self._get_deno_dir()
            if deno_dir:
                allowed_read_paths.append(deno_dir)

            if self.enable_read_paths:
                allowed_read_paths.extend(str(p) for p in self.enable_read_paths)
            if self.enable_write_paths:
                allowed_read_paths.extend(str(p) for p in self.enable_write_paths)
            args.append(f"--allow-read={','.join(allowed_read_paths)}")

            self._env_arg = ""
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

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_deno_dir() -> str | None:
        if "DENO_DIR" in os.environ:
            return os.environ["DENO_DIR"]

        try:
            result = subprocess.run(
                ["deno", "info", "--json"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return info.get("denoDir")
        except Exception:
            logger.warning("Unable to find the Deno cache dir.")

        return None

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

    def _extract_parameters(self, fn: Callable) -> list[dict]:
        """Extract parameter info from a callable for sandbox registration."""
        sig = inspect.signature(fn)
        params = []
        for name, param in sig.parameters.items():
            p = {"name": name}
            # Only include type for simple types that work in function signatures
            # Complex types like Union, Optional, etc. are not included
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in SIMPLE_TYPES:
                    p["type"] = param.annotation.__name__
            if param.default != inspect.Parameter.empty:
                p["default"] = param.default
            params.append(p)
        return params

    def _register_tools(self) -> None:
        """Register tools and output fields with the sandbox."""
        if self._tools_registered:
            return

        # Build registration message with typed tool signatures
        msg = {}

        if self.tools:
            tools_info = []
            for name, fn in self.tools.items():
                tools_info.append({
                    "name": name,
                    "parameters": self._extract_parameters(fn)
                })
            msg["register_tools"] = tools_info

        if self.output_fields:
            msg["register_outputs"] = self.output_fields

        # Skip if nothing to register
        if not msg:
            self._tools_registered = True
            return

        self.deno_process.stdin.write(json.dumps(msg) + "\n")
        self.deno_process.stdin.flush()
        response_line = self.deno_process.stdout.readline().strip()
        if not response_line:
            raise CodeInterpreterError("No response when registering tools/outputs")
        response = json.loads(response_line)
        if "tools_registered" not in response and "outputs_registered" not in response:
            raise CodeInterpreterError(f"Unexpected response when registering: {response_line}")
        self._tools_registered = True

    def _handle_tool_call(self, request: dict) -> None:
        """Handle a tool call request from the sandbox."""
        request_id, tool_name = request["id"], request["name"]
        call_args = request.get("args", {})

        try:
            if tool_name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {tool_name}")
            result = self.tools[tool_name](*call_args.get("args", []), **call_args.get("kwargs", {}))
            is_json = isinstance(result, (list, dict))
            response = {"type": "tool_response", "id": request_id, "error": None,
                        "result": json.dumps(result) if is_json else str(result or ""),
                        "result_type": "json" if is_json else "string"}
        except Exception as e:
            response = {"type": "tool_response", "id": request_id, "result": None, "error": str(e)}

        self.deno_process.stdin.write(json.dumps(response) + "\n")
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
                raise CodeInterpreterError(install_instructions) from e

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        """Insert Python assignments for each variable at the top of the code."""
        for key in variables:
            if not key.isidentifier() or keyword.iskeyword(key):
                raise CodeInterpreterError(f"Invalid variable name: '{key}'")
        assignments = [f"{k} = {self._serialize_value(v)}" for k, v in variables.items()]
        return "\n".join(assignments) + "\n" + code if assignments else code

    def _serialize_value(self, value: Any) -> str:
        """Serialize a Python value to a string representation for injection."""
        if value is None:
            return "None"
        elif isinstance(value, str):
            return repr(value)
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        elif isinstance(value, tuple):
            return json.dumps(list(value))
        elif isinstance(value, set):
            try:
                return json.dumps(sorted(value))
            except TypeError:
                return json.dumps(list(value))
        else:
            raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        variables = variables or {}
        code = self._inject_variables(code, variables)
        self._ensure_deno_process()
        self._mount_files()
        self._register_tools()

        # Send the code as JSON
        input_data = json.dumps({"code": code})
        try:
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
        except BrokenPipeError:
            # If the process died, restart and try again once
            self._tools_registered = False
            self._ensure_deno_process()
            self._register_tools()
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()

        # Read and handle messages until we get the final output.
        # Loop is needed because tool calls require back-and-forth communication.
        while True:
            output_line = self.deno_process.stdout.readline().strip()
            if not output_line:
                # Possibly the subprocess died or gave no output
                err_output = self.deno_process.stderr.read()
                raise CodeInterpreterError(f"No output from Deno subprocess. Stderr: {err_output}")

            # Parse that line as JSON
            try:
                result = json.loads(output_line)
            except json.JSONDecodeError:
                # If not valid JSON, just return raw text
                result = {"output": output_line}

            # Handle tool call requests from sandbox
            if result.get("type") == "tool_call":
                self._handle_tool_call(result)
                continue

            # Handle errors based on errorType
            if "error" in result:
                error_msg = result["error"]
                error_type = result.get("errorType", "Sandbox Error")
                error_args = result.get("errorArgs", [])

                if error_type == "FinalOutput":
                    output = error_args[0] if error_args else None
                    self._sync_files()
                    return FinalOutput(output)
                elif error_type == "SyntaxError":
                    raise SyntaxError(f"Invalid Python syntax. message: {error_msg}")
                else:
                    raise CodeInterpreterError(f"{error_type}: {error_args or error_msg}")

            # If there's no error or got `FinalAnswer`, return the "output" field
            self._sync_files()
            return result.get("output", None)

    def start(self) -> None:
        """Initialize the Deno/Pyodide sandbox.

        This pre-warms the sandbox by starting the Deno subprocess.
        Can be called explicitly for pooling, or will be called lazily
        on first execute().

        Idempotent: safe to call multiple times.
        """
        self._ensure_deno_process()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    def __call__(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        return self.execute(code, variables)

    def shutdown(self) -> None:
        if self.deno_process and self.deno_process.poll() is None:
            self.deno_process.stdin.write(json.dumps({"shutdown": True}) + "\n")
            self.deno_process.stdin.flush()
            self.deno_process.stdin.close()
            self.deno_process.wait()
        self.deno_process = None
