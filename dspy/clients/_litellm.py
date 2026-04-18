"""litellm backend module.

Implements the DSPy backend protocol using litellm. Every backend
(_litellm, _openai, _anthropic, …) exports the same interface so
LM can swap between them transparently.

This is the **only** module that imports litellm directly.
"""

import logging
import os
from typing import Any

import litellm

import dspy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# litellm configuration — called once on first use
# ---------------------------------------------------------------------------

_configured = False


def _configure_litellm():
    """One-time litellm setup: disable telemetry, cache, and noisy logging."""
    global _configured
    if _configured:
        return
    _configured = True

    litellm.telemetry = False
    litellm.cache = None  # DSPy has its own cache.
    disable_litellm_logging()


def configure_litellm_logging(level: str = "ERROR"):
    """Configure litellm logging to the specified level."""
    from litellm._logging import verbose_logger

    numeric_level = getattr(logging, level)
    verbose_logger.setLevel(numeric_level)
    for h in verbose_logger.handlers:
        h.setLevel(numeric_level)


def enable_litellm_logging():
    """Enable verbose litellm debug logging."""
    litellm.suppress_debug_info = False
    configure_litellm_logging("DEBUG")


def disable_litellm_logging():
    """Suppress litellm logging for clean output."""
    litellm.suppress_debug_info = True
    configure_litellm_logging("ERROR")

# ---------------------------------------------------------------------------
# Re-export the context-window error so callers can catch it generically
# ---------------------------------------------------------------------------

ContextWindowError = litellm.ContextWindowExceededError

# Always disable litellm's internal cache — DSPy has its own.
_NO_LITELLM_CACHE = {"no-cache": True, "no-store": True}

# ---------------------------------------------------------------------------
# Capability queries
#
# These take just a model string. The provider prefix is extracted
# internally so callers don't need to know about litellm's
# custom_llm_provider kwarg.
# ---------------------------------------------------------------------------


def _provider_prefix(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else "openai"


def supports_function_calling(model: str) -> bool:
    return litellm.supports_function_calling(model=model)


def supports_reasoning(model: str) -> bool:
    return litellm.supports_reasoning(model)


def supports_response_schema(model: str) -> bool:
    return litellm.supports_response_schema(model=model, custom_llm_provider=_provider_prefix(model))


def supported_params(model: str) -> set[str]:
    params = litellm.get_supported_openai_params(model=model, custom_llm_provider=_provider_prefix(model))
    return set(params) if params else set()


# ---------------------------------------------------------------------------
# Dispatching entry points
#
# These select the right completion function based on model_type and
# handle litellm cache disabling, so callers only pass
# (request, model_type, num_retries).
# ---------------------------------------------------------------------------


def complete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Sync completion — dispatches by model_type."""
    if model_type == "chat":
        return complete(request, num_retries)
    elif model_type == "text":
        return text_complete(request, num_retries)
    elif model_type == "responses":
        return responses_complete(request, num_retries)
    raise ValueError(f"Unknown model_type: {model_type!r}")


async def acomplete_request(request: dict[str, Any], model_type: str, num_retries: int):
    """Async completion — dispatches by model_type."""
    if model_type == "chat":
        return await acomplete(request, num_retries)
    elif model_type == "text":
        return await atext_complete(request, num_retries)
    elif model_type == "responses":
        return await aresponses_complete(request, num_retries)
    raise ValueError(f"Unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# Sync completion functions
# ---------------------------------------------------------------------------


def complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or _NO_LITELLM_CACHE
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    return litellm.completion(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=headers,
        **request,
    )


def text_complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or _NO_LITELLM_CACHE
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
    cache = cache or _NO_LITELLM_CACHE
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
    cache = cache or _NO_LITELLM_CACHE
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    return await litellm.acompletion(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def atext_complete(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or _NO_LITELLM_CACHE
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
    cache = cache or _NO_LITELLM_CACHE
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


# Re-export shared helpers for backward compat (tests import from here)
from dspy.clients._request_utils import convert_chat_to_responses_request as _convert_chat_request_to_responses_request  # noqa: F401
from dspy.clients._request_utils import _convert_content_item as _convert_content_item_to_responses_format  # noqa: F401


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None):
    headers = headers or {}
    return {"User-Agent": f"DSPy/{dspy.__version__}", **headers}


# ---------------------------------------------------------------------------
# New-protocol streaming (used by LM's unified streaming path)
# ---------------------------------------------------------------------------

from dspy.clients._request_utils import StreamChunk  # noqa: E402


def normalize_chunk(chunk) -> StreamChunk:
    """Convert a litellm ``ModelResponseStream`` to a ``StreamChunk``."""
    delta = chunk.choices[0].delta if chunk.choices else None
    return StreamChunk(
        content=getattr(delta, "content", None) if delta else None,
        reasoning_content=getattr(delta, "reasoning_content", None) if delta else None,
        tool_calls=[tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in delta.tool_calls] if delta and getattr(delta, "tool_calls", None) else None,
        provider_specific_fields=getattr(delta, "provider_specific_fields", None) if delta else None,
        finish_reason=chunk.choices[0].finish_reason if chunk.choices and chunk.choices[0].finish_reason else None,
    )


async def astream_complete(request: dict[str, Any], num_retries: int):
    """Return an async iterator of ``StreamChunk`` for litellm chat completion.

    After exhaustion, ``.assembled`` holds the rebuilt ``ModelResponse``.
    """
    cache = _NO_LITELLM_CACHE
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))

    if dspy.settings.track_usage:
        request["stream_options"] = {"include_usage": True}

    response = await litellm.acompletion(
        cache=cache,
        stream=True,
        headers=headers,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        **request,
    )
    return _LitellmStreamWrapper(response)


class _LitellmStreamWrapper:
    """Wraps a litellm async stream, normalizing chunks and collecting them."""

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
            self.assembled = litellm.stream_chunk_builder(self._raw_chunks)
            raise
        self._raw_chunks.append(raw)
        return normalize_chunk(raw)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def compute_embedding(model: str, inputs: list[str], caching: bool = False, **kwargs) -> list:
    """Compute embeddings for a batch of inputs via litellm."""
    _configure_litellm()
    caching = caching and litellm.cache is not None
    response = litellm.embedding(model=model, input=inputs, caching=caching, **kwargs)
    return [data["embedding"] for data in response.data]


async def acompute_embedding(model: str, inputs: list[str], caching: bool = False, **kwargs) -> list:
    """Async compute embeddings for a batch of inputs via litellm."""
    _configure_litellm()
    caching = caching and litellm.cache is not None
    response = await litellm.aembedding(model=model, input=inputs, caching=caching, **kwargs)
    return [data["embedding"] for data in response.data]
