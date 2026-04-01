"""Shared utilities for backend modules.

Contains request transformers, retry logic, and other helpers used
across multiple backends so each backend stays DRY.
"""

import asyncio
import time
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
