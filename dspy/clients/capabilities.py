"""Adapter-facing capability hints for native and custom execution engines.

The catalog describes models; lm15's policies describe the selected API.
A serializer's ability to send a field alone is not evidence a model honours it.
Public booleans preserve BaseLM's existing unknown-as-false convention.
"""

from dataclasses import dataclass, field

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


async def prepare_async(lm):
    """Prepare native model metadata before synchronous adapter planning.

    No provider clients or credentials are initialized. Legacy/custom engines
    keep their own capability contract; they are not made thread-safe here.
    """
    if getattr(lm, "_engine_spec", None) not in ("auto", "lm15") or lm.model_type == "text":
        return
    from dspy.clients.model_metadata import apreload
    from dspy.utils.exceptions import LMError

    try:
        resolve(lm)
    except LMError:
        return
    await apreload()


def capabilities(lm):
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
    if lm.model_type == "text":
        return Capabilities()
    from dspy.utils.exceptions import LMError

    try:
        route = resolve(lm)
    except LMError:
        return Capabilities()
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
