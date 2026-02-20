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
import threading
from os import PathLike
from typing import Any, Callable

from dspy.primitives.code_interpreter import SIMPLE_TYPES, CodeInterpreterError, FinalOutput

__all__ = ["PythonInterpreter", "FinalOutput", "CodeInterpreterError"]

logger = logging.getLogger(__name__)

# Pyodide's FFI crashes at exactly 128MB (134,217,728 bytes). Use filesystem
# injection for strings above 100MB to stay safely below this limit.
LARGE_VAR_THRESHOLD = 100 * 1024 * 1024

# =============================================================================
# JSON-RPC 2.0 Helpers
# =============================================================================

# JSON-RPC 2.0 protocol errors (reserved range: -32700 to -32600)
JSONRPC_PROTOCOL_ERRORS = {
    "ParseError": -32700,
    "InvalidRequest": -32600,
    "MethodNotFound": -32601,
}

# Application errors (range: -32000 to -32099)
JSONRPC_APP_ERRORS = {
    "SyntaxError": -32000,
    "NameError": -32001,
    "TypeError": -32002,
    "ValueError": -32003,
    "AttributeError": -32004,
    "IndexError": -32005,
    "KeyError": -32006,
    "RuntimeError": -32007,
    "CodeInterpreterError": -32008,
    "Unknown": -32099,
}


def _jsonrpc_request(method: str, params: dict, id: int | str) -> str:
    """Create a JSON-RPC 2.0 request (expects response)."""
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": id})


def _jsonrpc_notification(method: str, params: dict | None = None) -> str:
    """Create a JSON-RPC 2.0 notification (no response expected)."""
    msg = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    return json.dumps(msg)


def _jsonrpc_result(result: Any, id: int | str) -> str:
    """Create a JSON-RPC 2.0 success response."""
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": id})


def _jsonrpc_error(code: int, message: str, id: int | str, data: dict | None = None) -> str:
    """Create a JSON-RPC 2.0 error response."""
    err = {"code": code, "message": message}
    if data:
        err["data"] = data
    return json.dumps({"jsonrpc": "2.0", "error": err, "id": id})


class _SubprocessDied(Exception):
    """Internal: raised when the Deno subprocess dies during execution."""


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
        self._request_id = 0
        self._owner_thread: int | None = None
        self._pending_large_vars = {}

    def _check_thread_ownership(self) -> None:
        """Ensure this interpreter is only used from a single thread."""
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "PythonInterpreter is not thread-safe and cannot be shared across threads. "
                "Create a separate interpreter instance for each thread."
            )

    def _kill_process(self) -> None:
        """Terminate the subprocess and clear the handle to allow restart."""
        if self.deno_process is not None:
            try:
                self.deno_process.kill()
            except OSError:
                pass
            try:
                self.deno_process.wait(timeout=5)
            except Exception:
                pass
            self.deno_process = None

    def _write_msg(self, msg: str, context: str) -> None:
        """Write a JSON-RPC message to the subprocess. Raises _SubprocessDied on failure."""
        try:
            self.deno_process.stdin.write(msg + "\n")
            self.deno_process.stdin.flush()
        except BrokenPipeError:
            exit_code = self.deno_process.poll()
            raise _SubprocessDied(f"Deno subprocess died during {context} (exit code: {exit_code})")

    def _read_line(self, context: str) -> str:
        """Read a line from the subprocess. Raises _SubprocessDied on EOF."""
        line = self.deno_process.stdout.readline().strip()
        if not line:
            exit_code = self.deno_process.poll()
            stderr = self.deno_process.stderr.read() if self.deno_process.stderr else ""
            parts = [f"Deno subprocess produced no output during {context} (exit code: {exit_code})"]
            if stderr.strip():
                parts.append(f"Stderr: {stderr.strip()[:500]}")
            raise _SubprocessDied(". ".join(parts))
        return line

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
            self._send_request("mount_file", {"host_path": str(path), "virtual_path": virtual_path}, f"mounting {path}")
        self._mounted_files = True

    def _sync_files(self):
        if not self.enable_write_paths or not self.sync_files:
            return
        for path in self.enable_write_paths:
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            sync_msg = _jsonrpc_notification("sync_file", {"virtual_path": virtual_path, "host_path": str(path)})
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

        # Build registration params with typed tool signatures
        params = {}

        if self.tools:
            tools_info = []
            for name, fn in self.tools.items():
                tools_info.append({
                    "name": name,
                    "parameters": self._extract_parameters(fn)
                })
            params["tools"] = tools_info

        if self.output_fields:
            params["outputs"] = self.output_fields

        # Skip if nothing to register
        if not params:
            self._tools_registered = True
            return

        self._send_request("register", params, "registering tools/outputs")
        self._tools_registered = True

    def _handle_tool_call(self, request: dict) -> None:
        """Handle a tool call request from the sandbox."""
        request_id = request["id"]
        params = request.get("params", {})
        tool_name = params.get("name")
        args = params.get("args", [])
        kwargs = params.get("kwargs", {})

        try:
            if tool_name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {tool_name}")
            result = self.tools[tool_name](*args, **kwargs)
            is_json = isinstance(result, (list, dict))
            response = _jsonrpc_result(
                {"value": json.dumps(result) if is_json else str(result or ""), "type": "json" if is_json else "string"},
                request_id
            )
        except Exception as e:
            error_type = type(e).__name__
            error_code = JSONRPC_APP_ERRORS.get(error_type, JSONRPC_APP_ERRORS["Unknown"])
            response = _jsonrpc_error(error_code, str(e), request_id, {"type": error_type})

        self._write_msg(response, "tool call response")

    def _ensure_deno_process(self) -> None:
        if self.deno_process is None or self.deno_process.poll() is not None:
            self._tools_registered = False
            self._mounted_files = False
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
            self._health_check()

    def _send_request(self, method: str, params: dict, context: str) -> dict:
        """Send a JSON-RPC request and return the parsed response."""
        self._request_id += 1
        request_id = self._request_id
        self._write_msg(_jsonrpc_request(method, params, request_id), context)
        response = json.loads(self._read_line(context))
        if response.get("id") != request_id:
            raise CodeInterpreterError(f"Response ID mismatch {context}: expected {request_id}, got {response.get('id')}")
        if "error" in response:
            raise CodeInterpreterError(f"Error {context}: {response['error'].get('message', 'Unknown error')}")
        return response

    def _health_check(self) -> None:
        """Verify the subprocess is alive by executing a simple expression."""
        response = self._send_request("execute", {"code": "print(1+1)"}, "during health check")
        if response.get("result", {}).get("output", "").strip() != "2":
            raise CodeInterpreterError(f"Unexpected ping response: {response}")

    def _to_json_compatible(self, value: Any) -> Any:
        """Recursively convert Python values to JSON-compatible types."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, dict):
            return {k: self._to_json_compatible(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._to_json_compatible(v) for v in value]
        elif isinstance(value, set):
            try:
                return sorted(self._to_json_compatible(v) for v in value)
            except TypeError:
                return [self._to_json_compatible(v) for v in value]
        else:
            raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        """Insert Python assignments for each variable at the top of the code."""
        for key in variables:
            if not key.isidentifier() or keyword.iskeyword(key) or key == "json":
                raise CodeInterpreterError(f"Invalid variable name: '{key}'")

        large_vars = {}
        small_assignments = []
        for k, v in variables.items():
            serialized = self._serialize_value(v)
            if len(serialized) > LARGE_VAR_THRESHOLD:
                large_vars[k] = json.dumps(self._to_json_compatible(v))
            else:
                small_assignments.append(f"{k} = {serialized}")

        self._pending_large_vars = large_vars

        if large_vars:
            large_assignments = [f"{k} = json.loads(open('/tmp/dspy_vars/{k}.json').read())" for k in large_vars]
            assignments = ["import json"] + small_assignments + large_assignments
        else:
            assignments = small_assignments

        return "\n".join(assignments) + "\n" + code if assignments else code

    def _serialize_value(self, value: Any) -> str:
        """Serialize a Python value to a Python literal string for injection.

        Sets and tuples are converted to lists for JSON round-trip compatibility,
        since the sandbox returns values via JSON which doesn't support these types.
        """
        if value is None:
            return "None"
        elif isinstance(value, str):
            return repr(value)
        elif isinstance(value, bool):
            # Must check bool before int since bool is a subclass of int
            return "True" if value else "False"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            # Tuples become lists for JSON compatibility
            items = ", ".join(self._serialize_value(item) for item in value)
            return f"[{items}]"
        elif isinstance(value, dict):
            items = ", ".join(
                f"{self._serialize_value(k)}: {self._serialize_value(v)}"
                for k, v in value.items()
            )
            return f"{{{items}}}"
        elif isinstance(value, set):
            # Sets become sorted lists (or unsorted if mixed types) for JSON compatibility
            try:
                sorted_items = sorted(value)
            except TypeError:
                sorted_items = list(value)
            items = ", ".join(self._serialize_value(item) for item in sorted_items)
            return f"[{items}]"
        else:
            raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")

    def _inject_large_var(self, name: str, value: str) -> None:
        """Inject a large variable via the virtual filesystem."""
        self._send_request("inject_var", {"name": name, "value": value}, f"injecting variable '{name}'")

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        self._check_thread_ownership()
        variables = variables or {}
        code = self._inject_variables(code, variables)

        try:
            return self._execute_inner(code)
        except _SubprocessDied:
            self._kill_process()

        try:
            return self._execute_inner(code)
        except _SubprocessDied as e:
            raise CodeInterpreterError(f"Deno subprocess failed after automatic restart: {e}")

    def _execute_inner(self, code: str) -> Any:
        """Run code in the subprocess. Raises _SubprocessDied on process-death conditions."""
        self._ensure_deno_process()
        self._mount_files()
        self._register_tools()

        for name, value in self._pending_large_vars.items():
            self._inject_large_var(name, value)

        # Send the code as JSON-RPC request
        self._request_id += 1
        execute_request_id = self._request_id
        self._write_msg(_jsonrpc_request("execute", {"code": code}, execute_request_id), "execute")

        # Read and handle messages until we get the final output.
        # Loop is needed because tool calls require back-and-forth communication.
        while True:
            output_line = self._read_line("execute")

            # Skip non-JSON lines (e.g., Pyodide package loading messages)
            if not output_line.startswith("{"):
                logger.debug(f"Skipping non-JSON output: {output_line}")
                continue

            # Parse that line as JSON
            try:
                msg = json.loads(output_line)
            except json.JSONDecodeError:
                # Malformed JSON starting with '{' - log and continue
                logger.info(f"Skipping malformed JSON: {output_line[:100]}")
                continue

            # Handle incoming requests (tool calls from sandbox)
            if "method" in msg:
                if msg["method"] == "tool_call":
                    self._handle_tool_call(msg)
                    continue

            # Handle success response
            if "result" in msg:
                if msg.get("id") != execute_request_id:
                    raise CodeInterpreterError(f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}")
                result = msg["result"]
                self._sync_files()
                # Check for SUBMIT (encoded as success with "final" field)
                if "final" in result:
                    return FinalOutput(result["final"])
                return result.get("output", None)

            # Handle error response
            if "error" in msg:
                # Errors with id=null are unsolicited errors (e.g., unhandled async rejections)
                # Treat them as errors for the current request
                if msg.get("id") is not None and msg.get("id") != execute_request_id:
                    raise CodeInterpreterError(f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}")
                error = msg["error"]
                error_code = error.get("code", JSONRPC_APP_ERRORS["Unknown"])
                error_message = error.get("message", "Unknown error")
                error_data = error.get("data", {})
                error_type = error_data.get("type", "Error")

                if error_code == JSONRPC_APP_ERRORS["SyntaxError"]:
                    raise SyntaxError(f"Invalid Python syntax. message: {error_message}")
                else:
                    raise CodeInterpreterError(f"{error_type}: {error_data.get('args') or error_message}")

            # Unexpected message format - neither a recognized method nor a response
            raise CodeInterpreterError(f"Unexpected message format from sandbox: {msg}")

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
            self.deno_process.stdin.write(_jsonrpc_notification("shutdown") + "\n")
            self.deno_process.stdin.flush()
            self.deno_process.stdin.close()
            self.deno_process.wait()
        self.deno_process = None
        self._owner_thread = None
