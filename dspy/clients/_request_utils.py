"""Shared utilities for backend modules.

Contains request transformers, retry logic, streaming chunk types,
and other helpers used across multiple backends so each backend stays DRY.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import dspy


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def dspy_user_agent() -> str:
    return f"DSPy/{dspy.__version__}"


def strip_prefix(model: str) -> str:
    """Remove the provider prefix: 'openai/gpt-4o' → 'gpt-4o'."""
    return model.split("/", 1)[1] if "/" in model else model


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------


def call_with_retries(fn, num_retries: int, transient_errors: tuple, **kwargs):
    """Call *fn* with exponential backoff on transient errors."""
    last_err = None
    for attempt in range(num_retries + 1):
        try:
            return fn(**kwargs)
        except transient_errors as e:
            last_err = e
            if attempt < num_retries:
                time.sleep(2 ** attempt)
    raise last_err


async def acall_with_retries(fn, num_retries: int, transient_errors: tuple, **kwargs):
    """Async call *fn* with exponential backoff on transient errors."""
    last_err = None
    for attempt in range(num_retries + 1):
        try:
            return await fn(**kwargs)
        except transient_errors as e:
            last_err = e
            if attempt < num_retries:
                await asyncio.sleep(2 ** attempt)
    raise last_err


# ---------------------------------------------------------------------------
# StreamChunk — normalized streaming chunk for all backends
# ---------------------------------------------------------------------------


@dataclass
class StreamChunk:
    """A normalized streaming chunk produced by any DSPy backend.

    This is the common currency for DSPy's streaming system. Each backend
    converts its SDK-native chunk type into ``StreamChunk`` so that
    ``streamify`` and ``StreamListener`` never depend on a specific SDK.

    The shape mirrors the litellm ``ModelResponseStream`` / OpenAI
    ``ChatCompletionChunk`` layout so that ``StreamListener`` can access
    ``chunk.choices[0].delta.content`` without changes. Fields that are
    not present in a given chunk are ``None``.
    """

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    provider_specific_fields: dict | None = None
    finish_reason: str | None = None
    # Routing metadata set by the streaming loop.
    predict_id: int | None = None

    # ------------------------------------------------------------------
    # Compatibility shim — StreamListener accesses
    #   chunk.choices[0].delta.content  /  .reasoning_content  /  .provider_specific_fields
    # We expose an attribute-access façade so existing listener code works.
    # ------------------------------------------------------------------

    @property
    def choices(self):
        return [_ChoiceShim(self)]

    def json(self):
        """Serialize to JSON string (used by ``streaming_response``)."""
        d: dict[str, Any] = {}
        if self.content is not None:
            d["content"] = self.content
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.provider_specific_fields is not None:
            d["provider_specific_fields"] = self.provider_specific_fields
        if self.finish_reason is not None:
            d["finish_reason"] = self.finish_reason
        return json.dumps(d)


@dataclass
class _DeltaShim:
    """Attribute-access shim mimicking ``delta.content``, etc."""
    _chunk: "StreamChunk"

    @property
    def content(self):
        return self._chunk.content

    @property
    def reasoning_content(self):
        return self._chunk.reasoning_content

    @property
    def tool_calls(self):
        return self._chunk.tool_calls

    @property
    def provider_specific_fields(self):
        return self._chunk.provider_specific_fields


@dataclass
class _ChoiceShim:
    """Attribute-access shim mimicking ``choices[0].delta``."""
    _chunk: "StreamChunk"

    @property
    def delta(self):
        return _DeltaShim(self._chunk)


# ---------------------------------------------------------------------------
# Request transformation: chat → Responses API
# ---------------------------------------------------------------------------


def convert_chat_to_responses_request(request: dict[str, Any]) -> dict[str, Any]:
    """Convert a chat-format request to the OpenAI Responses API format.

    See https://platform.openai.com/docs/api-reference/responses/create
    """
    import pydantic

    request = dict(request)
    if "messages" in request:
        content_blocks = []
        for msg in request.pop("messages"):
            c = msg.get("content")
            if isinstance(c, str):
                content_blocks.append({"type": "input_text", "text": c})
            elif isinstance(c, list):
                for item in c:
                    content_blocks.append(_convert_content_item(item))
        request["input"] = [{"role": msg.get("role", "user"), "content": content_blocks}]

    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}

    if "response_format" in request:
        response_format = request.pop("response_format")
        if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
            response_format = {
                "name": response_format.__name__,
                "type": "json_schema",
                "schema": response_format.model_json_schema(),
            }
        text = request.pop("text", {})
        request["text"] = {**text, "format": response_format}

    return request


def _convert_content_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat API content item to Responses API format."""
    if item.get("type") == "image_url":
        image_url = item.get("image_url", {}).get("url", "")
        return {"type": "input_image", "image_url": image_url}
    elif item.get("type") == "text":
        return {"type": "input_text", "text": item.get("text", "")}
    elif item.get("type") == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }
    return item
