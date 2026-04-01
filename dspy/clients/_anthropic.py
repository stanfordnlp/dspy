"""Anthropic SDK backend module.

Implements the DSPy backend protocol using the anthropic Python SDK.
Translates DSPy's OpenAI-shaped requests/responses to and from
the Anthropic Messages API.

No litellm dependency.
"""

import asyncio
import json
import logging
import time
from typing import Any

import anthropic
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

import dspy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context-window error
# ---------------------------------------------------------------------------

ContextWindowError = anthropic.BadRequestError

# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------


def supports_function_calling(model: str) -> bool:
    return True  # All Claude models support tool use


def supports_reasoning(model: str) -> bool:
    family = model.split("/")[-1] if "/" in model else model
    return "claude-3-7" in family or "claude-4" in family


def supports_response_schema(model: str) -> bool:
    return False  # Anthropic doesn't support OpenAI-style response_format


def supported_params(model: str) -> set[str]:
    return {
        "temperature", "max_tokens", "top_p", "top_k",
        "stop_sequences", "tools", "tool_choice", "stream",
    }


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def _make_client(request: dict, async_: bool = False):
    """Build an Anthropic client, popping auth keys from the request."""
    kwargs = {}
    if "api_key" in request:
        kwargs["api_key"] = request.pop("api_key")
    if "api_base" in request:
        kwargs["base_url"] = request.pop("api_base")
    elif "base_url" in request:
        kwargs["base_url"] = request.pop("base_url")
    cls = anthropic.AsyncAnthropic if async_ else anthropic.Anthropic
    return cls(**kwargs)


def _strip_prefix(model: str) -> str:
    """'anthropic/claude-3-5-sonnet' → 'claude-3-5-sonnet'."""
    return model.split("/", 1)[1] if "/" in model else model


# ---------------------------------------------------------------------------
# Request translation: OpenAI format → Anthropic format
# ---------------------------------------------------------------------------

# OpenAI params that have no Anthropic equivalent
_STRIP_PARAMS = {
    "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs",
    "response_format", "seed", "n", "parallel_tool_calls", "stream_options",
    "rollout_id", "headers", "max_completion_tokens",
}

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _translate_request(request: dict) -> dict:
    """Translate an OpenAI-shaped request dict into Anthropic's format."""
    request = dict(request)

    # Strip provider prefix
    request["model"] = _strip_prefix(request["model"])

    # Extract system message from the messages list
    messages = request.pop("messages", [])
    system_parts = []
    user_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg["content"])
        else:
            user_messages.append(msg)
    request["messages"] = user_messages
    if system_parts:
        request["system"] = "\n\n".join(system_parts)

    # max_tokens is required by Anthropic — default to 4096
    if "max_tokens" not in request and "max_completion_tokens" not in request:
        request["max_tokens"] = 4096
    elif "max_completion_tokens" in request:
        request["max_tokens"] = request.pop("max_completion_tokens")

    # Rename stop → stop_sequences
    if "stop" in request:
        stop = request.pop("stop")
        if stop is not None:
            request["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    # Translate reasoning_effort → thinking
    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        budget_map = {"low": 1024, "medium": 4096, "high": 16384}
        budget = budget_map.get(effort, 4096)
        request["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # Translate tools from OpenAI format to Anthropic format
    if "tools" in request:
        request["tools"] = [_translate_tool(t) for t in request["tools"]]

    # Strip params Anthropic doesn't understand
    for key in list(request.keys()):
        if key in _STRIP_PARAMS:
            request.pop(key)

    return request


def _translate_tool(tool: dict) -> dict:
    """Translate an OpenAI tool dict to Anthropic format."""
    if tool.get("type") == "function":
        func = tool["function"]
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        }
    return tool


# ---------------------------------------------------------------------------
# Response translation: Anthropic format → OpenAI format
# ---------------------------------------------------------------------------


def _translate_response(msg: anthropic.types.Message) -> ChatCompletion:
    """Translate an Anthropic Message into an OpenAI ChatCompletion."""
    text_parts = []
    reasoning_parts = []
    tool_calls = []

    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            reasoning_parts.append(block.thinking)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                },
            })

    content = "\n".join(text_parts) if text_parts else None

    message_kwargs = {"role": "assistant", "content": content}
    if reasoning_parts:
        # Attach as extra attribute — ChatCompletionMessage accepts arbitrary fields
        message_kwargs["reasoning_content"] = "\n".join(reasoning_parts)
    if tool_calls:
        message_kwargs["tool_calls"] = tool_calls

    message = ChatCompletionMessage(**message_kwargs)

    finish_reason = _STOP_REASON_MAP.get(msg.stop_reason, "stop")

    choice = Choice(finish_reason=finish_reason, index=0, message=message)

    usage = CompletionUsage(
        prompt_tokens=msg.usage.input_tokens,
        completion_tokens=msg.usage.output_tokens,
        total_tokens=msg.usage.input_tokens + msg.usage.output_tokens,
    )

    return ChatCompletion(
        id=msg.id,
        choices=[choice],
        created=0,
        model=msg.model,
        object="chat.completion",
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_TRANSIENT = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)


def _call_with_retries(fn, num_retries: int, **kwargs):
    last_err = None
    for attempt in range(num_retries + 1):
        try:
            return fn(**kwargs)
        except _TRANSIENT as e:
            last_err = e
            if attempt < num_retries:
                time.sleep(2 ** attempt)
    raise last_err


async def _acall_with_retries(fn, num_retries: int, **kwargs):
    last_err = None
    for attempt in range(num_retries + 1):
        try:
            return await fn(**kwargs)
        except _TRANSIENT as e:
            last_err = e
            if attempt < num_retries:
                await asyncio.sleep(2 ** attempt)
    raise last_err


# ---------------------------------------------------------------------------
# Dispatching entry points
# ---------------------------------------------------------------------------


def complete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Sync completion — only chat supported for Anthropic."""
    if model_type != "chat":
        raise ValueError(
            f"Anthropic backend only supports model_type='chat', got {model_type!r}."
        )
    return complete(request, num_retries)


async def acomplete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Async completion — only chat supported for Anthropic."""
    if model_type != "chat":
        raise ValueError(
            f"Anthropic backend only supports model_type='chat', got {model_type!r}."
        )
    return await acomplete(request, num_retries)


# ---------------------------------------------------------------------------
# Sync / async completion
# ---------------------------------------------------------------------------


def complete(request: dict[str, Any], num_retries: int):
    request = _translate_request(request)
    client = _make_client(request)
    msg = _call_with_retries(client.messages.create, num_retries, **request)
    return _translate_response(msg)


async def acomplete(request: dict[str, Any], num_retries: int):
    request = _translate_request(request)
    client = _make_client(request, async_=True)
    msg = await _acall_with_retries(client.messages.create, num_retries, **request)
    return _translate_response(msg)
