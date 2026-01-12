"""
E2B-based sandbox for secure remote code execution.

This module provides E2BSandbox, which runs Python code in a sandboxed
Firecracker microVM using E2B's Code Interpreter service. It implements the
Sandbox protocol defined in sandbox.py.

Prerequisites:
    E2B API key: Sign up at https://e2b.dev/ and set E2B_API_KEY environment variable
    Install: pip install e2b-code-interpreter
"""

import json
import logging
import os
from typing import Any, Callable

from dspy.primitives.sandbox import FinalAnswerResult, SandboxError

__all__ = ["E2BSandbox"]

logger = logging.getLogger(__name__)


class E2BSandbox:
    """Remote sandbox using E2B Code Interpreter.

    Implements the Sandbox protocol for secure code execution in E2B's
    cloud-based Firecracker microVM sandbox. Code runs in an isolated environment
    with no access to the host filesystem or network (unless explicitly enabled).

    Prerequisites:
        Set E2B_API_KEY environment variable or pass api_key to constructor.
        Install: pip install e2b-code-interpreter

    Features:
        - Fully isolated execution in Firecracker microVM
        - State persists across execute() calls (Jupyter kernel)
        - ~150ms startup time
        - No host filesystem access (safer than local sandbox)

    Example:
        ```python
        # Basic execution
        with E2BSandbox() as sandbox:
            result = sandbox.execute("print(1 + 2)")  # Returns "3"

        # With variables
        with E2BSandbox() as sandbox:
            result = sandbox.execute("print(x + y)", variables={"x": 10, "y": 20})
        ```
    """

    # Setup code run once per session to define FINAL/FINAL_VAR
    _SETUP_CODE = '''
class FinalAnswer(BaseException):
    """Control-flow exception to signal completion with a result.

    This pattern uses exceptions because:
    1. FINAL() can be called from any nesting depth
    2. Exceptions propagate naturally across execution boundaries
    3. No need to thread return values through every call
    """
    pass

def FINAL(answer):
    """Signal completion and return the given answer."""
    raise FinalAnswer(answer)

def FINAL_VAR(var_name):
    """Signal completion and return the value of the named variable."""
    if var_name in globals():
        raise FinalAnswer(globals()[var_name])
    raise NameError(f"Variable '{var_name}' not found")
'''

    # Code to set up llm_query using litellm in the sandbox
    _LLM_SETUP_CODE = '''
import os
import litellm

# NOTE: Hardcoded to openai
_llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
_llm_call_count = 0
_llm_max_calls = int(os.environ.get("LLM_MAX_CALLS", "50"))

def llm_query(prompt: str = "") -> str:
    """Query the LLM with a prompt string."""
    global _llm_call_count
    if not prompt:
        raise ValueError("prompt is required")
    if _llm_call_count >= _llm_max_calls:
        raise RuntimeError(f"LLM call limit exceeded: {_llm_call_count} >= {_llm_max_calls}")
    _llm_call_count += 1
    response = litellm.completion(
        model=_llm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def llm_query_batched(prompts: list) -> list:
    """Query the LLM with multiple prompts (sequentially for now)."""
    return [llm_query(p) for p in prompts]
'''

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 300,
        tools: dict[str, Callable[..., str]] | None = None,
        openai_api_key: str | None = None,
        llm_model: str = "gpt-4o-mini",
        max_llm_calls: int = 50,
    ) -> None:
        """
        Args:
            api_key: E2B API key. Defaults to E2B_API_KEY environment variable.
            timeout: Sandbox timeout in seconds (default 5 minutes, max 24 hours).
            tools: Dictionary mapping tool names to callable functions.
                   Note: Custom tool support requires additional setup.
            openai_api_key: OpenAI API key to enable llm_query() in sandbox.
                           If provided, llm_query and llm_query_batched will be available.
                           Defaults to OPENAI_API_KEY environment variable.
            llm_model: Model to use for llm_query. Defaults to gpt-4o-mini.
            max_llm_calls: Maximum number of LLM calls allowed per session.
        """
        self._api_key = api_key or os.environ.get("E2B_API_KEY")
        self._timeout = timeout
        self._tools = tools or {}
        self._sandbox = None
        self._setup_done = False

        # LLM configuration for llm_query support
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._llm_model = llm_model
        self._max_llm_calls = max_llm_calls

        if self._tools:
            logger.warning(
                "E2BSandbox custom tool support is not yet implemented. "
                "Tools will be ignored. llm_query is available if openai_api_key is set."
            )

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        """Tools available for sandbox code to call."""
        return self._tools

    def _ensure_sandbox(self) -> None:
        """Create sandbox if not already running."""
        if self._sandbox is not None:
            return

        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as e:
            raise ImportError(
                "e2b-code-interpreter is required for E2BSandbox. "
                "Install with: pip install e2b-code-interpreter"
            ) from e

        if not self._api_key:
            raise SandboxError(
                "E2B API key required. Set E2B_API_KEY environment variable "
                "or pass api_key to E2BSandbox constructor. "
                "Sign up at https://e2b.dev/ to get an API key."
            )

        # Build environment variables for the sandbox
        envs = {}
        if self._openai_api_key:
            envs["OPENAI_API_KEY"] = self._openai_api_key
            envs["LLM_MODEL"] = self._llm_model
            envs["LLM_MAX_CALLS"] = str(self._max_llm_calls)

        self._sandbox = Sandbox.create(
            api_key=self._api_key,
            timeout=self._timeout,
            envs=envs if envs else None,
        )

    def _run_setup(self) -> None:
        """Run one-time setup code to define FINAL/FINAL_VAR and optionally llm_query."""
        if self._setup_done:
            return

        self._ensure_sandbox()

        try:
            # Setup FINAL/FINAL_VAR
            execution = self._sandbox.run_code(self._SETUP_CODE)
            if execution.error:
                err_msg = getattr(execution.error, "value", None) or str(execution.error)
                raise SandboxError(f"Setup failed: {err_msg}")

            # Setup llm_query if OpenAI API key is available
            if self._openai_api_key:
                # Install litellm in the sandbox
                execution = self._sandbox.run_code("import subprocess; subprocess.run(['pip', 'install', '-q', 'litellm'], check=True)")
                if execution.error:
                    err_msg = getattr(execution.error, "value", None) or str(execution.error)
                    logger.warning(f"Failed to install litellm: {err_msg}")
                else:
                    execution = self._sandbox.run_code(self._LLM_SETUP_CODE)
                    if execution.error:
                        err_msg = getattr(execution.error, "value", None) or str(execution.error)
                        logger.warning(f"LLM setup failed (llm_query won't be available): {err_msg}")

            self._setup_done = True
        except Exception as e:
            # If setup fails due to timeout, reset and re-raise
            error_str = str(e).lower()
            if "timeout" in error_str or "not found" in error_str or "502" in str(e):
                logger.warning(f"Sandbox timed out during setup, will recreate on next call: {e}")
                self._reset_sandbox()
            raise

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        """Insert Python assignments for each variable at the top of the code."""
        if not variables:
            return code

        for key in variables:
            if not key.isidentifier():
                raise SandboxError(f"Invalid variable name: '{key}'")

        assignments = [f"{k} = {self._serialize_value(v)}" for k, v in variables.items()]
        return "\n".join(assignments) + "\n" + code

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
        else:
            raise SandboxError(f"Unsupported value type: {type(value).__name__}")

    def _extract_final_answer(self, execution) -> Any:
        """Extract the FinalAnswer value from an execution error.

        When FinalAnswer is raised, E2B returns it as an error with the
        answer available in error.value.
        """
        error_value = getattr(execution.error, "value", None)
        if error_value is not None:
            return error_value
        return getattr(execution.error, "message", None)

    def _reset_sandbox(self) -> None:
        """Reset sandbox state to force recreation on next use."""
        self._sandbox = None
        self._setup_done = False

    def _run_code_with_reconnect(self, code: str) -> Any:
        """Run code with automatic reconnection on timeout."""
        try:
            return self._sandbox.run_code(code)
        except Exception as e:
            # Check for timeout/sandbox not found errors
            error_str = str(e).lower()
            if "timeout" in error_str or "not found" in error_str or "502" in str(e):
                logger.warning(f"Sandbox timed out, recreating: {e}")
                self._reset_sandbox()
                self._run_setup()
                return self._sandbox.run_code(code)
            raise

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code and return the result.

        Args:
            code: Python code to execute
            variables: Variables to inject into the namespace before execution.
                      These are available as top-level variables in the code.

        Returns:
            One of:
            - FinalAnswerResult: If FINAL() or FINAL_VAR() was called in code
            - str: Captured stdout from print() statements
            - None: If no output was produced

        Raises:
            SandboxError: On runtime errors (undefined vars, tool failures, etc.)
            SyntaxError: On invalid Python syntax
        """
        self._run_setup()

        variables = variables or {}
        code = self._inject_variables(code, variables)

        execution = self._run_code_with_reconnect(code)

        # Check for errors
        if execution.error:
            error_name = getattr(execution.error, "name", "Error")
            error_message = getattr(execution.error, "message", str(execution.error))

            if error_name == "FinalAnswer":
                answer = self._extract_final_answer(execution)
                return FinalAnswerResult(answer)
            elif error_name == "SyntaxError":
                raise SyntaxError(f"Invalid Python syntax: {error_message}")
            else:
                raise SandboxError(f"{error_name}: {error_message}")

        # Collect output: prefer stdout from logs, fall back to text (expression result)
        output_parts = []
        if execution.logs and execution.logs.stdout:
            output_parts.extend(execution.logs.stdout)
        if execution.text:
            output_parts.append(execution.text)

        if output_parts:
            return "".join(output_parts).rstrip("\n")
        return None

    def start(self) -> None:
        """Initialize the E2B sandbox.

        This pre-warms the sandbox by creating the E2B Code Interpreter session.
        Can be called explicitly for pooling, or will be called lazily
        on first execute().

        Idempotent: safe to call multiple times.
        """
        self._ensure_sandbox()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    def __call__(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Alias for execute() to allow calling the sandbox directly."""
        return self.execute(code, variables)

    def shutdown(self) -> None:
        """Release resources and terminate the sandbox.

        After shutdown(), the sandbox should not be used again.
        A new instance should be created for a fresh session.
        """
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception as e:
                logger.warning(f"Error killing E2B sandbox: {e}")
            self._sandbox = None
            self._setup_done = False
