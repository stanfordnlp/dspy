"""litellm backend module.

Stateless functions that wrap litellm for DSPy's backend protocol.
All litellm imports and configuration live here — nowhere else in dspy/clients/.

Functions use generic names (complete, acomplete, etc.) so callers never
reference "litellm" directly.
"""

import logging
import os
from typing import Any, cast

import litellm
from anyio.streams.memory import MemoryObjectSendStream
from asyncer import syncify

import dspy
from dspy.dsp.utils.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export the context-window error so callers can catch it generically
# ---------------------------------------------------------------------------

ContextWindowError = litellm.ContextWindowExceededError

# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------


def supports_function_calling(model: str) -> bool:
    return litellm.supports_function_calling(model=model)


def supports_reasoning(model: str) -> bool:
    return litellm.supports_reasoning(model)


def supports_response_schema(model: str, custom_llm_provider: str | None = None) -> bool:
    return litellm.supports_response_schema(model=model, custom_llm_provider=custom_llm_provider)


def get_supported_params(model: str, custom_llm_provider: str | None = None) -> set[str]:
    params = litellm.get_supported_openai_params(model=model, custom_llm_provider=custom_llm_provider)
    return set(params) if params else set()


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


def _get_stream_completion_fn(
    request: dict[str, Any],
    cache_kwargs: dict[str, Any],
    sync: bool = True,
    headers: dict[str, Any] | None = None,
):
    stream = dspy.settings.send_stream
    caller_predict = dspy.settings.caller_predict

    if stream is None:
        return None

    stream = cast(MemoryObjectSendStream, stream)
    caller_predict_id = id(caller_predict) if caller_predict else None

    if dspy.settings.track_usage:
        request["stream_options"] = {"include_usage": True}

    async def stream_completion(request: dict[str, Any], cache_kwargs: dict[str, Any]):
        response = await litellm.acompletion(
            cache=cache_kwargs,
            stream=True,
            headers=headers,
            **request,
        )
        chunks = []
        async for chunk in response:
            if caller_predict_id:
                chunk.predict_id = caller_predict_id
            chunks.append(chunk)
            await stream.send(chunk)
        return litellm.stream_chunk_builder(chunks)

    def sync_stream_completion():
        syncified_stream_completion = syncify(stream_completion)
        return syncified_stream_completion(request, cache_kwargs)

    async def async_stream_completion():
        return await stream_completion(request, cache_kwargs)

    if sync:
        return sync_stream_completion
    else:
        return async_stream_completion


# ---------------------------------------------------------------------------
# Sync completion functions
# ---------------------------------------------------------------------------


def complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    stream_completion = _get_stream_completion_fn(request, cache, sync=True, headers=headers)
    if stream_completion is None:
        return litellm.completion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            headers=headers,
            **request,
        )

    return stream_completion()


def text_complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return litellm.text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def responses_complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)

    return litellm.responses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


# ---------------------------------------------------------------------------
# Async completion functions
# ---------------------------------------------------------------------------


async def acomplete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    stream_completion = _get_stream_completion_fn(request, cache, sync=False)
    if stream_completion is None:
        return await litellm.acompletion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            headers=_add_dspy_identifier_to_headers(headers),
            **request,
        )

    return await stream_completion()


async def atext_complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    model = request.pop("model").split("/", 1)
    headers = request.pop("headers", None)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return await litellm.atext_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def aresponses_complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)

    return await litellm.aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


# ---------------------------------------------------------------------------
# Request transformation helpers
# ---------------------------------------------------------------------------


def _convert_chat_request_to_responses_request(request: dict[str, Any]):
    """Convert a chat request to a responses request.

    See https://platform.openai.com/docs/api-reference/responses/create for the responses API specification.
    Also see https://platform.openai.com/docs/api-reference/chat/create for the chat API specification.
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
                    content_blocks.append(_convert_content_item_to_responses_format(item))
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


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a content item from Chat API format to Responses API format."""
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


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None):
    headers = headers or {}
    return {"User-Agent": f"DSPy/{dspy.__version__}", **headers}
