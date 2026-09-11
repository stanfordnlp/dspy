"""Adapter-facing capability hints for native and custom execution engines.

The catalog describes models; lm15's policies describe the selected API.
A serializer's ability to send a field alone is not evidence a model honours it.
Public booleans preserve BaseLM's existing unknown-as-false convention.
"""

import asyncio
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

from dspy.clients.backend_selection import select_backend
from dspy.clients.engines.litellm_errors import litellm_errors
from dspy.clients.errors import error_boundary
from dspy.clients.model_metadata import model_info


@dataclass(frozen=True)
class Capabilities:
    function_calling: bool = False
    reasoning: bool = False
    response_schema: bool = False
    params: frozenset[str] = field(default_factory=frozenset)


def resolve(lm):
    from dspy.clients.engines.lm15_engine import LM15Engine
    from dspy.lm15 import RouterConfig

    return LM15Engine(RouterConfig(env={}), model_type=lm.model_type).resolve(lm.model)


@dataclass(repr=False)
class _PlanningScope:
    lm: object
    options: dict
    value: Capabilities | None = None


_planning = ContextVar("dspy_lm_capability_planning", default=None)


@contextmanager
def planning_scope(lm, options):
    current = _planning.get()
    if current is not None and current.lm is lm and current.options is options:
        yield
        return
    token = _planning.set(_PlanningScope(lm, options))
    try:
        yield
    finally:
        _planning.reset(token)


def with_capability_planning(fn):
    """Scope adapter capability reads to this call's client settings.

    Do not copy or mutate the LM: callbacks and custom hooks retain its identity.
    Nested JSON/base-adapter calls reuse the same task-local planning scope.
    """
    if inspect.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_wrapper(self, lm, lm_kwargs, *args, **kwargs):
            with planning_scope(lm, lm_kwargs):
                await prepare_async(lm)
                return await fn(self, lm, lm_kwargs, *args, **kwargs)
        return async_wrapper

    @wraps(fn)
    def wrapper(self, lm, lm_kwargs, *args, **kwargs):
        with planning_scope(lm, lm_kwargs):
            return fn(self, lm, lm_kwargs, *args, **kwargs)
    return wrapper


async def prepare_async(lm):
    """Resolve built-in capability hints off-loop, including LiteLLM lookup.

    Native calls still never import LiteLLM. Custom/legacy LM capability hooks
    keep their own contract and are not moved to worker threads.
    """
    scope = _planning.get()
    if scope is not None and scope.lm is lm and scope.value is not None:
        return
    if isinstance(getattr(lm, "_engine_spec", None), str):
        await asyncio.to_thread(capabilities, lm)


def capabilities(lm):
    with error_boundary(lm.model, unexpected=True):
        scope = _planning.get()
        if scope is not None and scope.lm is lm:
            if scope.value is None:
                scope.value = _capabilities(lm, scope.options)
            return scope.value
        return _capabilities(lm)


def _litellm_capabilities(lm, selection):
    from dspy.clients.lm import _get_litellm

    litellm = _get_litellm()
    provider = selection.clients.get("custom_llm_provider")
    kwargs = {"custom_llm_provider": provider} if provider is not None else {}
    with litellm_errors(model=lm.model, provider=provider or lm._provider_name):
        params = litellm.get_supported_openai_params(model=lm.model, custom_llm_provider=provider or lm._provider_name)
        return Capabilities(
            bool(litellm.supports_function_calling(model=lm.model, **kwargs)),
            bool(litellm.supports_reasoning(lm.model, **kwargs)),
            bool(litellm.supports_response_schema(model=lm.model, custom_llm_provider=provider or lm._provider_name)),
            frozenset(params or ()),
        )


def _capabilities(lm, options=None):
    spec = lm.engine
    if not isinstance(spec, str):
        # A custom engine can declare the same hints as BaseLM. Missing
        # declarations remain false; do not consult an unrelated model catalog.
        return Capabilities(
            bool(getattr(spec, "supports_function_calling", False)),
            bool(getattr(spec, "supports_reasoning", False)),
            bool(getattr(spec, "supports_response_schema", False)),
            frozenset(getattr(spec, "supported_params", ()) or ()),
        )
    selection = select_backend(lm, options)
    if not selection.native:
        return _litellm_capabilities(lm, selection)
    route = selection.resolution
    from dspy._vendor.lm15.compat import (
        AnthropicCompat,
        OpenAIChatCompat,
        OpenAIResponsesCompat,
        resolve_anthropic_compat,
        resolve_openai_chat_compat,
        resolve_openai_responses_compat,
    )
    from dspy._vendor.lm15.registry import lookup

    definition = lookup(route.provider)
    info = model_info(route.provider, route.model)
    tools = info.get("supports_function_calling") is True
    reasoning = info.get("supports_reasoning") is True
    schema = info.get("supports_response_schema") is True
    params = {"temperature", "max_tokens", "top_p", "stream"}
    # These describe lm15's canonical mappings. Engine-specific refusal still
    # validates the complete request, including combinations and model limits.
    if definition.dialect == "openai-chat":
        preset = definition.compat or ("xai" if route.provider == "xai" else "openai")
        compat = resolve_openai_chat_compat(OpenAIChatCompat.preset(preset).for_model(route.model))
        params.update({"stop", "max_completion_tokens", "logprobs", "top_logprobs", "seed",
                       "presence_penalty", "frequency_penalty", "logit_bias", "user"})
        if info.get("supports_response_schema") is True or info.get("supports_function_calling") is True:
            params.add("response_format")
        if compat.json_schema == "reject":
            schema = False
        if compat.thinking_format == "none":
            reasoning = False
        if route.provider == "xai":
            params.difference_update({"logprobs", "top_logprobs"})
    elif definition.dialect == "anthropic":
        compat = resolve_anthropic_compat(AnthropicCompat.preset(definition.compat or "anthropic"))
        params.update({"stop", "top_k"})
        if schema and compat.structured_output != "reject":
            params.add("response_format")
        else:
            schema = False
        if compat.sampling_params == "reject":
            params.difference_update({"temperature", "top_p", "top_k"})
    elif definition.dialect == "gemini":
        params.update({"stop", "top_k", "logprobs", "top_logprobs", "response_format"})
    else:
        compat = resolve_openai_responses_compat(OpenAIResponsesCompat.preset(definition.compat or "openai"))
        params.update({"response_format", "max_completion_tokens", "logprobs", "top_logprobs", "store"})
        if compat.reasoning_format == "none":
            reasoning = False
        if route.provider == "openai-codex":
            params.difference_update({"max_tokens", "max_completion_tokens", "store"})
    if tools:
        params.update({"tools", "tool_choice", "parallel_tool_calls"})
        if definition.dialect == "gemini" or (
            definition.dialect == "anthropic" and compat.parallel_tool_calls == "reject"
        ):
            params.discard("parallel_tool_calls")
    if reasoning:
        params.add("reasoning_effort")
    return Capabilities(tools, reasoning, schema, frozenset(params))
