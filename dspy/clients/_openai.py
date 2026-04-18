"""OpenAI SDK backend module.

Implements the DSPy backend protocol using the openai Python SDK directly.
Covers openai/* and azure/* models, plus any OpenAI-compatible server
reached via api_base (vLLM, Ollama, SGLang, Together, Arbor, etc.).

Supports chat completions, the Responses API, and legacy text completions.
No litellm dependency.
"""

import logging
import os
import re
from typing import Any

import openai

from dspy.clients._request_utils import (
    StreamChunk,
    acall_with_retries,
    call_with_retries,
    convert_chat_to_responses_request,
    dspy_user_agent,
    strip_prefix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context-window error
# ---------------------------------------------------------------------------

ContextWindowError = openai.BadRequestError

# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------

_FUNCTION_CALLING = re.compile(r"(gpt-4|gpt-3\.5-turbo|gpt-5|o[1345])")
_REASONING = re.compile(r"^(o[1345]|gpt-5)(-|$)")
_RESPONSE_SCHEMA = re.compile(r"(gpt-4o|gpt-5|o[1345])")


def _model_family(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def supports_function_calling(model: str) -> bool:
    return bool(_FUNCTION_CALLING.search(_model_family(model)))


def supports_reasoning(model: str) -> bool:
    return bool(_REASONING.match(_model_family(model)))


def supports_response_schema(model: str) -> bool:
    return bool(_RESPONSE_SCHEMA.search(_model_family(model)))


def supported_params(model: str) -> set[str]:
    return {
        "temperature", "max_tokens", "max_completion_tokens", "top_p",
        "frequency_penalty", "presence_penalty", "stop", "n", "logprobs",
        "top_logprobs", "response_format", "seed", "tools", "tool_choice",
        "parallel_tool_calls", "stream", "stream_options", "reasoning_effort",
    }


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def _make_client(request: dict, async_: bool = False):
    """Build an OpenAI client, popping auth/base keys from the request."""
    kwargs = {}
    if "api_key" in request:
        kwargs["api_key"] = request.pop("api_key")
    if "api_base" in request:
        kwargs["base_url"] = request.pop("api_base")
    elif "base_url" in request:
        kwargs["base_url"] = request.pop("base_url")
    kwargs["default_headers"] = {"User-Agent": dspy_user_agent()}
    cls = openai.AsyncOpenAI if async_ else openai.OpenAI
    return cls(**kwargs)


def _prepare(request: dict) -> dict:
    """Clean the request dict for the OpenAI SDK."""
    request = dict(request)
    request.pop("rollout_id", None)
    request.pop("headers", None)
    request["model"] = strip_prefix(request["model"])
    return request


_TRANSIENT = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


# ---------------------------------------------------------------------------
# Dispatching entry points
# ---------------------------------------------------------------------------


def complete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Sync completion — dispatches by model_type."""
    if model_type == "chat":
        return chat_complete(request, num_retries)
    elif model_type == "text":
        return text_complete(request, num_retries)
    elif model_type == "responses":
        return responses_complete(request, num_retries)
    raise ValueError(f"Unknown model_type: {model_type!r}")


async def acomplete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Async completion — dispatches by model_type."""
    if model_type == "chat":
        return await achat_complete(request, num_retries)
    elif model_type == "text":
        return await atext_complete(request, num_retries)
    elif model_type == "responses":
        return await aresponses_complete(request, num_retries)
    raise ValueError(f"Unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def chat_complete(request: dict[str, Any], num_retries: int):
    request = _prepare(request)
    client = _make_client(request)
    return call_with_retries(client.chat.completions.create, num_retries, _TRANSIENT, **request)


async def achat_complete(request: dict[str, Any], num_retries: int):
    request = _prepare(request)
    client = _make_client(request, async_=True)
    return await acall_with_retries(client.chat.completions.create, num_retries, _TRANSIENT, **request)


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


def responses_complete(request: dict[str, Any], num_retries: int):
    request = _prepare(request)
    request = convert_chat_to_responses_request(request)
    client = _make_client(request)
    return call_with_retries(client.responses.create, num_retries, _TRANSIENT, **request)


async def aresponses_complete(request: dict[str, Any], num_retries: int):
    request = _prepare(request)
    request = convert_chat_to_responses_request(request)
    client = _make_client(request, async_=True)
    return await acall_with_retries(client.responses.create, num_retries, _TRANSIENT, **request)


# ---------------------------------------------------------------------------
# Legacy text completions
# ---------------------------------------------------------------------------


def text_complete(request: dict[str, Any], num_retries: int):
    request = _prepare(request)
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])
    request.pop("model")
    client = _make_client(request)
    return call_with_retries(client.completions.create, num_retries, _TRANSIENT, prompt=prompt, **request)


async def atext_complete(request: dict[str, Any], num_retries: int):
    request = _prepare(request)
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])
    request.pop("model")
    client = _make_client(request, async_=True)
    return await acall_with_retries(client.completions.create, num_retries, _TRANSIENT, prompt=prompt, **request)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def normalize_chunk(chunk) -> StreamChunk:
    """Convert an OpenAI ``ChatCompletionChunk`` to a ``StreamChunk``."""
    delta = chunk.choices[0].delta if chunk.choices else None
    return StreamChunk(
        content=getattr(delta, "content", None) if delta else None,
        reasoning_content=getattr(delta, "reasoning_content", None) if delta else None,
        tool_calls=[tc.model_dump() for tc in delta.tool_calls] if delta and delta.tool_calls else None,
        finish_reason=chunk.choices[0].finish_reason if chunk.choices else None,
    )


def _assemble_chat_chunks(chunks) -> openai.types.chat.ChatCompletion:
    """Reassemble collected ``ChatCompletionChunk`` objects into a ``ChatCompletion``."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types import CompletionUsage

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_idx: dict[int, dict] = {}
    finish_reason = "stop"
    model = ""
    chunk_id = ""
    prompt_tokens = 0
    completion_tokens = 0

    for chunk in chunks:
        if not chunk.choices:
            # usage-only chunk
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            continue
        delta = chunk.choices[0].delta
        if chunk.model:
            model = chunk.model
        if chunk.id:
            chunk_id = chunk.id
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if delta.content:
            text_parts.append(delta.content)
        if getattr(delta, "reasoning_content", None):
            reasoning_parts.append(delta.reasoning_content)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_by_idx:
                    tool_calls_by_idx[idx] = {
                        "id": tc.id or "",
                        "type": "function",
                        "function": {"name": tc.function.name or "", "arguments": ""},
                    }
                else:
                    if tc.id:
                        tool_calls_by_idx[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_by_idx[idx]["function"]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls_by_idx[idx]["function"]["arguments"] += tc.function.arguments

    content = "".join(text_parts) if text_parts else None
    msg_kwargs: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_parts:
        msg_kwargs["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls_by_idx:
        msg_kwargs["tool_calls"] = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]

    message = ChatCompletionMessage(**msg_kwargs)
    choice = Choice(finish_reason=finish_reason, index=0, message=message)
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return ChatCompletion(
        id=chunk_id or "stream",
        choices=[choice],
        created=0,
        model=model,
        object="chat.completion",
        usage=usage,
    )


async def astream_complete(request: dict[str, Any], num_retries: int):
    """Return an async iterator of ``StreamChunk`` for chat completions.

    Yields normalized ``StreamChunk`` objects. After the iterator is
    exhausted, its ``.assembled`` attribute holds the reassembled
    ``ChatCompletion``.
    """
    request = _prepare(request)
    request["stream"] = True
    request["stream_options"] = {"include_usage": True}
    client = _make_client(request, async_=True)
    stream = await acall_with_retries(client.chat.completions.create, num_retries, _TRANSIENT, **request)
    return _OpenAIStreamWrapper(stream)


class _OpenAIStreamWrapper:
    """Wraps an OpenAI async stream, normalizing chunks and collecting them."""

    def __init__(self, stream):
        self._stream = stream
        self._raw_chunks: list = []
        self.assembled = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamChunk:
        try:
            raw = await self._stream.__anext__()
        except StopAsyncIteration:
            self.assembled = _assemble_chat_chunks(self._raw_chunks)
            raise
        self._raw_chunks.append(raw)
        return normalize_chunk(raw)
