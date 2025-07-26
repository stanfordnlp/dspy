import json
import logging
import subprocess
from typing import Any, Literal

from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)


class ClaudeCodeLM(BaseLM):
    """
    Claude Code LM client that uses the Claude Code CLI for direct LLM access.
    
    This provides actual LLM inference through Claude Code's CLI interface,
    unlike the MCP server which only provides tools.
    """

    def __init__(
        self,
        model: str = "claude-code",
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        max_turns: int = 1,
        cache: bool = True,
        claude_model: str | None = None,
        **kwargs,
    ):
        """
        Initialize Claude Code LM client.

        Args:
            model: Model identifier for this instance
            model_type: The type of model ("chat" or "text")
            temperature: Sampling temperature (not directly supported by Claude CLI)
            max_tokens: Maximum tokens to generate (not directly supported by Claude CLI)
            max_turns: Maximum agentic turns for Claude Code
            cache: Whether to cache responses
            claude_model: Specific Claude model ("sonnet", "opus", etc.)
            **kwargs: Additional arguments
        """
        super().__init__(model=model, model_type=model_type, temperature=temperature, max_tokens=max_tokens, cache=cache, **kwargs)

        self.max_turns = max_turns
        self.claude_model = claude_model

        # Verify Claude Code is available
        self._verify_claude_code()

    def _verify_claude_code(self):
        """Verify Claude Code CLI is installed and available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Claude Code CLI available: {result.stdout.strip()}")
            else:
                raise FileNotFoundError("Claude Code CLI not working")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            ) from e

    def _build_claude_command(self, prompt: str) -> list[str]:
        """Build Claude CLI command with appropriate options."""
        cmd = ["claude", "-p", prompt, "--output-format", "json"]

        if self.claude_model:
            cmd.extend(["--model", self.claude_model])

        if self.max_turns != 1:
            cmd.extend(["--max-turns", str(self.max_turns)])

        return cmd

    def _execute_claude_command(self, cmd: list[str]) -> dict[str, Any]:
        """Execute Claude CLI command and parse JSON response."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for LLM responses
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Claude CLI command failed"
                raise RuntimeError(f"Claude CLI error: {error_msg}")

            # Parse JSON response
            if result.stdout.strip():
                return json.loads(result.stdout)
            else:
                raise RuntimeError("Claude CLI returned empty response")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI command timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Claude CLI JSON response: {e}")

    def _convert_claude_response_to_openai(self, claude_response: dict[str, Any]) -> object:
        """Convert Claude CLI response to OpenAI-compatible format."""

        # Extract content from Claude CLI JSON response
        # Expected format: {"type":"result","result":"content",...}
        content = ""
        if isinstance(claude_response, dict):
            if "result" in claude_response and claude_response.get("type") == "result":
                content = claude_response["result"]
            elif "content" in claude_response:
                content = claude_response["content"]
            elif "message" in claude_response:
                content = claude_response["message"]
            elif "text" in claude_response:
                content = claude_response["text"]
            else:
                # Fallback: try to extract any string value
                for key, value in claude_response.items():
                    if isinstance(value, str) and key not in ["type", "subtype", "session_id"]:
                        content = value
                        break
        elif isinstance(claude_response, str):
            content = claude_response
        else:
            content = str(claude_response)

        return self._create_openai_response(content, claude_response)

    def _create_openai_response(self, content: str, claude_response: dict = None) -> object:
        """Create OpenAI-compatible response object."""
        class Choice:
            def __init__(self, content: str):
                self.message = Message(content)
                self.finish_reason = "stop"

        class Message:
            def __init__(self, content: str):
                self.content = content

        class Usage:
            def __init__(self, claude_response: dict = None):
                # Extract real token usage from Claude CLI response
                if claude_response and "usage" in claude_response:
                    usage = claude_response["usage"]
                    self.prompt_tokens = usage.get("input_tokens", 0)
                    self.completion_tokens = usage.get("output_tokens", 0)
                    self.total_tokens = self.prompt_tokens + self.completion_tokens
                else:
                    # Fallback: estimate from content
                    self.prompt_tokens = len(content.split()) if content else 0
                    self.completion_tokens = len(content.split()) if content else 0
                    self.total_tokens = self.prompt_tokens + self.completion_tokens

            def __iter__(self):
                """Make Usage object iterable for DSPy compatibility."""
                return iter([
                    ("prompt_tokens", self.prompt_tokens),
                    ("completion_tokens", self.completion_tokens),
                    ("total_tokens", self.total_tokens)
                ])

        class Response:
            def __init__(self, content: str, model: str, claude_response: dict = None):
                self.choices = [Choice(content)]
                self.usage = Usage(claude_response)
                self.model = model

        return Response(content, self.model, claude_response)

    def forward(self, prompt=None, messages=None, **kwargs):
        """Synchronous forward pass using Claude CLI."""
        # Convert messages to a single prompt if needed
        if messages and not prompt:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt = "\n\n".join(prompt_parts)
        elif not prompt:
            raise ValueError("Either prompt or messages must be provided")

        # Build and execute Claude command
        cmd = self._build_claude_command(prompt)
        logger.debug(f"Executing Claude CLI command: {cmd}")

        try:
            claude_response = self._execute_claude_command(cmd)
            return self._convert_claude_response_to_openai(claude_response)
        except Exception as e:
            logger.error(f"Claude CLI request failed: {e}")
            return self._create_error_response(str(e))

    def _create_error_response(self, error_msg: str) -> object:
        """Create error response in OpenAI format."""
        error_content = f"Claude Code Error: {error_msg}"
        return self._create_openai_response(error_content)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        """Async forward pass (runs sync command in thread)."""
        import asyncio
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.forward, prompt, messages, **kwargs)
            return await asyncio.wrap_future(future)


def create_claude_code_lm(**kwargs) -> ClaudeCodeLM:
    """
    Create Claude Code LM instance for direct LLM access.
    
    Args:
        **kwargs: Arguments for ClaudeCodeLM constructor
        
    Returns:
        ClaudeCodeLM instance ready for DSPy
        
    Examples:
        ```python
        import dspy
        from dspy.clients.claude_code_lm import create_claude_code_lm
        
        # Create Claude Code LM instance
        lm = create_claude_code_lm(claude_model="sonnet", max_turns=3)
        
        # Use with DSPy
        dspy.configure(lm=lm)
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="What is machine learning?")
        ```
    """
    return ClaudeCodeLM(**kwargs)
