"""Google GenAI SDK backend module.

Implements the DSPy backend protocol using the google-genai Python SDK.
Translates DSPy's OpenAI-shaped requests/responses to and from
the Gemini generateContent API.

No litellm dependency.
"""

import json
import logging
from typing import Any

import google.genai as genai
from google.genai import errors as genai_errors
from google.genai import types
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

from dspy.clients._request_utils import (
    StreamChunk,
    acall_with_retries,
    call_with_retries,
    strip_prefix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context-window error
# ---------------------------------------------------------------------------

ContextWindowError = genai_errors.ClientError

# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------


def supports_function_calling(model: str) -> bool:
    return True  # All Gemini models support function calling


def supports_reasoning(model: str) -> bool:
    family = _model_family(model)
    return "2.5" in family  # Gemini 2.5 models support thinking


def supports_response_schema(model: str) -> bool:
    return True  # Gemini supports structured output via responseSchema


def supported_params(model: str) -> set[str]:
    return {
        "temperature", "max_tokens", "top_p", "top_k",
        "stop", "n", "seed", "frequency_penalty", "presence_penalty",
        "response_format", "tools", "tool_choice", "logprobs",
        "reasoning_effort",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_family(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model





_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}


# ---------------------------------------------------------------------------
# Request translation: OpenAI format → Google GenAI format
# ---------------------------------------------------------------------------

_STRIP_PARAMS = {
    "rollout_id", "headers", "parallel_tool_calls", "stream_options",
    "max_completion_tokens",
}


def _translate_request(request: dict) -> tuple[str, list, types.GenerateContentConfig]:
    """Translate an OpenAI-shaped request into Google GenAI arguments.

    Returns (model, contents, config) ready for client.models.generate_content().
    """
    request = dict(request)
    model = strip_prefix(request.pop("model"))

    # Strip params Google doesn't understand
    for key in list(request.keys()):
        if key in _STRIP_PARAMS:
            request.pop(key)

    # Extract and translate messages → contents
    messages = request.pop("messages", [])
    contents, system_instruction = _translate_messages(messages)

    # Build GenerateContentConfig from remaining params
    config_kwargs = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    if "temperature" in request:
        config_kwargs["temperature"] = request.pop("temperature")
    if "max_tokens" in request:
        config_kwargs["max_output_tokens"] = request.pop("max_tokens")
    if "top_p" in request:
        config_kwargs["top_p"] = request.pop("top_p")
    if "top_k" in request:
        config_kwargs["top_k"] = request.pop("top_k")
    if "stop" in request:
        stop = request.pop("stop")
        if stop is not None:
            config_kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    if "n" in request:
        config_kwargs["candidate_count"] = request.pop("n")
    if "seed" in request:
        config_kwargs["seed"] = request.pop("seed")
    if "frequency_penalty" in request:
        config_kwargs["frequency_penalty"] = request.pop("frequency_penalty")
    if "presence_penalty" in request:
        config_kwargs["presence_penalty"] = request.pop("presence_penalty")

    # Response format
    if "response_format" in request:
        response_format = request.pop("response_format")
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                config_kwargs["response_mime_type"] = "application/json"
            elif response_format.get("type") == "json_schema":
                config_kwargs["response_mime_type"] = "application/json"
                if "json_schema" in response_format:
                    config_kwargs["response_json_schema"] = response_format["json_schema"].get("schema", {})

    # Reasoning / thinking
    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        budget_map = {"low": 1024, "medium": 4096, "high": 16384}
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=budget_map.get(effort, 4096)
        )

    # Logprobs
    if "logprobs" in request:
        logprobs = request.pop("logprobs")
        if logprobs:
            config_kwargs["response_logprobs"] = True
    if "top_logprobs" in request:
        config_kwargs["logprobs"] = request.pop("top_logprobs")

    # Tools
    if "tools" in request:
        config_kwargs["tools"] = [_translate_tool(t) for t in request.pop("tools")]
    if "tool_choice" in request:
        tool_choice = request.pop("tool_choice")
        if isinstance(tool_choice, str):
            mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=mode_map.get(tool_choice, "AUTO")
                )
            )

    # Pop remaining OpenAI-specific params we haven't handled
    request.pop("api_key", None)
    request.pop("api_base", None)
    request.pop("base_url", None)

    config = types.GenerateContentConfig(**config_kwargs)
    return model, contents, config


def _translate_messages(messages: list[dict]) -> tuple[list, str | None]:
    """Translate OpenAI messages to Google Content objects.

    Returns (contents, system_instruction).
    """
    system_parts = []
    contents = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            system_parts.append(msg["content"])
            continue

        # Map roles: user→user, assistant→model
        google_role = "model" if role == "assistant" else "user"
        parts = _translate_content(msg.get("content", ""))
        contents.append(types.Content(role=google_role, parts=parts))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return contents, system_instruction


def _translate_content(content) -> list[types.Part]:
    """Translate message content to Google Parts."""
    if isinstance(content, str):
        return [types.Part(text=content)]

    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(types.Part(text=item["text"]))
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Base64 inline image
                    import base64
                    header, data = url.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                    parts.append(types.Part(
                        inline_data=types.Blob(mime_type=mime, data=base64.b64decode(data))
                    ))
                else:
                    # URL reference — pass as file_data
                    parts.append(types.Part(
                        file_data=types.FileData(file_uri=url)
                    ))
        return parts

    return [types.Part(text=str(content))]


def _translate_tool(tool: dict) -> types.Tool:
    """Translate an OpenAI tool to Google format."""
    if tool.get("type") == "function":
        func = tool["function"]
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters"),
            )
        ])
    return types.Tool()


# ---------------------------------------------------------------------------
# Response translation: Google GenAI → OpenAI ChatCompletion
# ---------------------------------------------------------------------------


def _translate_response(resp: types.GenerateContentResponse, model: str) -> ChatCompletion:
    """Translate a Google GenAI response into an OpenAI ChatCompletion."""
    choices = []

    for i, candidate in enumerate(resp.candidates or []):
        text_parts = []
        reasoning_parts = []
        tool_calls = []

        for part in (candidate.content.parts if candidate.content else []):
            if part.function_call:
                fc = part.function_call
                tool_calls.append({
                    "id": f"call_{i}_{fc.name}",
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args) if isinstance(fc.args, dict) else str(fc.args),
                    },
                })
            elif part.thought:
                reasoning_parts.append(part.text or "")
            elif part.text is not None:
                text_parts.append(part.text)

        content = "\n".join(text_parts) if text_parts else None

        message_kwargs = {"role": "assistant", "content": content}
        if reasoning_parts:
            message_kwargs["reasoning_content"] = "\n".join(reasoning_parts)
        if tool_calls:
            message_kwargs["tool_calls"] = tool_calls

        message = ChatCompletionMessage(**message_kwargs)

        finish = candidate.finish_reason.name if candidate.finish_reason else "STOP"
        finish_reason = _FINISH_REASON_MAP.get(finish, "stop")

        choices.append(Choice(finish_reason=finish_reason, index=i, message=message))

    # Usage
    um = resp.usage_metadata
    usage = CompletionUsage(
        prompt_tokens=um.prompt_token_count or 0 if um else 0,
        completion_tokens=um.candidates_token_count or 0 if um else 0,
        total_tokens=um.total_token_count or 0 if um else 0,
    )

    return ChatCompletion(
        id=resp.response_id or "genai-response",
        choices=choices,
        created=0,
        model=model,
        object="chat.completion",
        usage=usage,
    )


_TRANSIENT = (genai_errors.ServerError,)


# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------


def _make_client(request: dict) -> genai.Client:
    """Build a Google GenAI client."""
    kwargs = {}
    if "api_key" in request:
        kwargs["api_key"] = request.pop("api_key")
    # api_base not standard for Google, but allow override
    request.pop("api_base", None)
    request.pop("base_url", None)
    return genai.Client(**kwargs)


# ---------------------------------------------------------------------------
# Dispatching entry points
# ---------------------------------------------------------------------------


def complete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Sync completion — only chat supported for Google backend."""
    if model_type != "chat":
        raise ValueError(
            f"Google backend only supports model_type='chat', got {model_type!r}."
        )
    return complete(request, num_retries)


async def acomplete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Async completion — only chat supported for Google backend."""
    if model_type != "chat":
        raise ValueError(
            f"Google backend only supports model_type='chat', got {model_type!r}."
        )
    return await acomplete(request, num_retries)


# ---------------------------------------------------------------------------
# Sync / async completion
# ---------------------------------------------------------------------------


def complete(request: dict[str, Any], num_retries: int):
    model, contents, config = _translate_request(request)
    client = _make_client(request)
    resp = call_with_retries(
        client.models.generate_content,
        num_retries,
        _TRANSIENT,
        model=model,
        contents=contents,
        config=config,
    )
    return _translate_response(resp, model)


async def acomplete(request: dict[str, Any], num_retries: int):
    model, contents, config = _translate_request(request)
    client = _make_client(request)
    resp = await acall_with_retries(
        client.aio.models.generate_content,
        num_retries,
        _TRANSIENT,
        model=model,
        contents=contents,
        config=config,
    )
    return _translate_response(resp, model)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def normalize_chunk(resp, model: str = "") -> StreamChunk:
    """Convert a Google GenAI streaming ``GenerateContentResponse`` to ``StreamChunk``."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = None

    for candidate in (resp.candidates or []):
        if candidate.finish_reason:
            finish_reason = _FINISH_REASON_MAP.get(candidate.finish_reason.name, "stop")
        for part in (candidate.content.parts if candidate.content else []):
            if part.function_call:
                fc = part.function_call
                tool_calls.append({
                    "id": f"call_{fc.name}",
                    "type": "function",
                    "function": {"name": fc.name, "arguments": json.dumps(fc.args) if isinstance(fc.args, dict) else str(fc.args)},
                })
            elif part.thought:
                reasoning_parts.append(part.text or "")
            elif part.text is not None:
                content_parts.append(part.text)

    return StreamChunk(
        content="".join(content_parts) if content_parts else None,
        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
        tool_calls=tool_calls or None,
        finish_reason=finish_reason,
    )


async def astream_complete(request: dict[str, Any], num_retries: int):
    """Return an async iterator of ``StreamChunk`` for Google GenAI.

    After exhaustion, ``.assembled`` holds the ``ChatCompletion``.
    """
    model, contents, config = _translate_request(request)
    client = _make_client(request)
    stream = await acall_with_retries(
        client.aio.models.generate_content_stream,
        num_retries,
        _TRANSIENT,
        model=model,
        contents=contents,
        config=config,
    )
    return _GoogleStreamWrapper(stream, model)


class _GoogleStreamWrapper:
    """Wraps a Google GenAI async stream, normalizing chunks."""

    def __init__(self, stream, model: str):
        self._stream = stream
        self._model = model
        self._raw_chunks: list = []
        self.assembled = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamChunk:
        try:
            raw = await self._stream.__anext__()
        except StopAsyncIteration:
            self.assembled = self._build_response()
            raise
        self._raw_chunks.append(raw)
        return normalize_chunk(raw, self._model)

    def _build_response(self):
        """Merge all raw chunks into a single ``ChatCompletion``."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict] = []
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        response_id = ""

        for resp in self._raw_chunks:
            if resp.response_id:
                response_id = resp.response_id
            um = resp.usage_metadata
            if um:
                prompt_tokens = um.prompt_token_count or prompt_tokens
                completion_tokens = um.candidates_token_count or completion_tokens
                total_tokens = um.total_token_count or total_tokens
            for candidate in (resp.candidates or []):
                if candidate.finish_reason:
                    finish_reason = _FINISH_REASON_MAP.get(candidate.finish_reason.name, "stop")
                for part in (candidate.content.parts if candidate.content else []):
                    if part.function_call:
                        fc = part.function_call
                        tool_calls.append({
                            "id": f"call_{fc.name}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args) if isinstance(fc.args, dict) else str(fc.args),
                            },
                        })
                    elif part.thought:
                        reasoning_parts.append(part.text or "")
                    elif part.text is not None:
                        text_parts.append(part.text)

        content = "".join(text_parts) if text_parts else None
        msg_kwargs: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_parts:
            msg_kwargs["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            msg_kwargs["tool_calls"] = tool_calls

        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types import CompletionUsage

        message = ChatCompletionMessage(**msg_kwargs)
        choice = Choice(finish_reason=finish_reason, index=0, message=message)
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return ChatCompletion(
            id=response_id or "genai-stream",
            choices=[choice],
            created=0,
            model=self._model,
            object="chat.completion",
            usage=usage,
        )
