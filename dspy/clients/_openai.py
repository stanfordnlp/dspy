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
