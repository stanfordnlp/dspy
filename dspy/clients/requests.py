"""The one place DSPy turns a rendered prompt and generation options into a Request.

Adapters render a `Prompt` (system text plus lm15 messages) and collect
generation options in the OpenAI vocabulary that `dspy.LM(...)` and
`dspy.Predict(config=...)` have always used (`temperature`, `max_tokens`,
`reasoning_effort`, `response_format`, ...). Those options are read with lm15's
own Chat Completions reader, so DSPy keeps no second copy of that vocabulary.
Everything past this boundary is an lm15 `Request`.

Three groups of keys never reach a Request:

- execution controls owned by the DSPy LM layer (`cache`, `rollout_id`, `n`);
- client settings owned by engines (`api_key`, `api_base`, ...);
- provider-specific options lm15 has no verdict for. They ride in
  `Config.extensions` and select the LiteLLM compatibility engine under
  `engine="auto"`; `engine="lm15"` refuses them before execution.
"""

from dataclasses import replace
from typing import Any

import pydantic

from dspy._vendor.lm15.providers import openai_chat as _openai_chat
from dspy.clients.backend_selection import CLIENT_KEYS, select_backend
from dspy.clients.legacy_requests import _resolve_ref
from dspy.lm15 import (
    BuiltinTool,
    CacheConfig,
    FunctionTool,
    Message,
    OpenAIChatCompat,
    Reasoning,
    Request,
    ToolChoice,
    UnsupportedFeatureError,
    request_from_openai_chat,
)
from dspy.utils.exceptions import LMUnsupportedFeatureError

EXECUTION_KEYS = frozenset({"cache", "rollout_id", "n", "num_generations"})
__all__ = ["CLIENT_KEYS", "EXECUTION_KEYS", "build_request", "generation_options", "passthrough_keys"]
_KNOWN_KEYS = (
    _openai_chat._INGEST_CONFIG_KEYS | _openai_chat._INGEST_EXTENSIONS_KEYS | _openai_chat._INGEST_CALL_MODE_KEYS
)
# Deprecated function-calling shapes are rejected outright; every other key
# lm15 refuses is a provider option only the compatibility engine can carry.
_REJECTED_KEYS = frozenset({"functions", "function_call"})
# Extension keys lm15 itself reads back from a Chat Completions body.
_CANONICAL_EXTENSION_KEYS = _openai_chat._INGEST_EXTENSIONS_KEYS | frozenset({"reasoning_format"})
_STUB_MESSAGES = [{"role": "user", "content": "."}]


def passthrough_keys(request: Request) -> frozenset[str]:
    """Extension keys lm15 has no canonical verdict for; they need LiteLLM."""
    return frozenset(key for key in (request.config.extensions or {}) if key not in _CANONICAL_EXTENSION_KEYS)


def close_object_schemas(schema: Any, *, root: dict | None = None) -> Any:
    """Shape generated schemas for strict structured outputs, as in DSPy 3.4."""
    if not isinstance(schema, dict):
        return schema
    root = schema if root is None else root
    closed = dict(schema)
    for key in ("items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"):
        if isinstance(closed.get(key), dict):
            closed[key] = close_object_schemas(closed[key], root=root)
    for key in ("items", "prefixItems", "anyOf", "oneOf", "allOf"):
        if isinstance(closed.get(key), list):
            closed[key] = [close_object_schemas(item, root=root) for item in closed[key]]
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        if isinstance(closed.get(key), dict):
            closed[key] = {name: close_object_schemas(sub, root=root) for name, sub in closed[key].items()}
    if closed.get("type") == "object" and "additionalProperties" not in closed:
        closed["additionalProperties"] = False
    if isinstance(closed.get("properties"), dict):
        closed["required"] = list(closed["properties"])
    if isinstance(closed.get("allOf"), list) and len(closed["allOf"]) == 1:
        closed.update(closed.pop("allOf")[0])
    if "default" in closed and closed["default"] is None:
        del closed["default"]
    ref = closed.get("$ref")
    if isinstance(ref, str) and len(closed) > 1:
        resolved = _resolve_ref(root, ref)
        if isinstance(resolved, dict):
            closed = {**close_object_schemas(resolved, root=root), **{k: v for k, v in closed.items() if k != "$ref"}}
    return closed


def generation_options(lm, options: dict | None = None) -> dict[str, Any]:
    """LM defaults overridden by call options; None removes a default."""
    merged = dict(lm.kwargs)
    for aliases in (("max_tokens", "max_completion_tokens"), ("reasoning", "reasoning_effort")):
        if options and any(name in options for name in aliases):
            # A call overrides the setting, regardless of its default spelling.
            for name in aliases:
                merged.pop(name, None)
    merged.update(options or {})
    return {
        key: value for key, value in merged.items()
        if key not in EXECUTION_KEYS and key not in CLIENT_KEYS and value is not None
    }


def build_request(lm, prompt, options: dict | None = None) -> Request:
    """Turn a rendered prompt and OpenAI-style options into the canonical Request for `lm`.

    Args:
        lm: The LM whose model and generation defaults apply.
        prompt: A `dspy.adapters.Prompt`, or a string rendered as one user message.
        options: Call-time generation options in the OpenAI vocabulary.
    """
    if isinstance(prompt, str):
        system, messages = None, (Message.user(prompt),)
    else:
        system, messages = prompt.system, tuple(prompt.messages)
    if getattr(lm, "use_developer_role", False) and lm.model_type == "responses" and isinstance(system, str):
        system, messages = None, (Message.developer(system), *messages)
    body = {"model": lm.model, "messages": _STUB_MESSAGES, **generation_options(lm, options)}
    if getattr(lm, "model_type", None) == "text" and type(body.get("logprobs")) is int:
        # Text endpoints encode the alternative count in logprobs itself.
        count = body["logprobs"]
        if "top_logprobs" in body and body["top_logprobs"] != count:
            raise ValueError("Text logprobs and top_logprobs specify different alternative counts")
        body.update(logprobs=True, top_logprobs=count)
    prompt_cache = body.pop("prompt_cache", None)
    if prompt_cache is not None and not isinstance(prompt_cache, CacheConfig):
        raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
    rejected = _REJECTED_KEYS.intersection(body)
    if rejected:
        raise LMUnsupportedFeatureError(
            f"{sorted(rejected)} use the deprecated function-calling shape; declare tools as "
            "dspy.lm15.FunctionTool objects and select them with tool_choice.",
            model=lm.model, features=sorted(rejected),
        )
    passthrough = {key: body.pop(key) for key in list(body) if key not in _KNOWN_KEYS}
    provider_controls = {"thinking", "enable_thinking", "chat_template_kwargs", "reasoning_format"}.intersection(body)
    selection = None
    if isinstance(lm.engine, str):
        try:
            selection = select_backend(lm, options)
        except Exception:
            # Preserve offline construction even for an unresolved native
            # route, unless provider-shaped controls require its verdict.
            if provider_controls:
                raise
    compat = None
    if selection is not None and selection.native and selection.resolution.adapter == "OpenAIChatLM":
        compat = selection.resolution.compat
        if isinstance(compat, str):
            compat = OpenAIChatCompat.preset(compat)
        compat = compat.for_model(lm.model) if compat is not None else None
    native_thinking = None
    if provider_controls and selection is not None:
        if not selection.native:
            # LiteLLM owns these provider spellings on the compatibility route.
            passthrough.update({key: body.pop(key) for key in provider_controls})
        elif selection.resolution.adapter == "AnthropicLM" and "thinking" in body:
            # The Messages writer already accepts this object via extensions.
            # Keep its budget/disable semantics out of the Chat-only reader.
            native_thinking = passthrough["thinking"] = body.pop("thinking")
    # The Responses API spells the reasoning dial as an object; DSPy has always
    # accepted that spelling on Responses models beside `reasoning_effort`.
    reasoning = body.pop("reasoning", None) if isinstance(body.get("reasoning"), dict) else None
    if reasoning is not None:
        body.pop("reasoning_effort", None)
    elif "reasoning_effort" in body and compat is not None and compat.thinking_format == "openrouter":
        # This wire spells the generic effort knob as an object. Other
        # presets retain their native combined-control/refusal validation.
        reasoning = {"effort": body.pop("reasoning_effort")}
    if reasoning is not None:
        effort = reasoning.get("effort", "medium")
        if reasoning.get("enabled") is False or effort == "none":
            effort = "off"
        reasoning = Reasoning(effort=effort, summary=reasoning.get("summary"))
        if "thinking" in body or (compat is not None and compat.thinking_format == "none"):
            # Keep native combined-control validation and explicit no-dial
            # refusal identical for dictionary and scalar effort spellings.
            body["reasoning_effort"] = "none" if reasoning.is_off else reasoning.effort
    tools = list(body.pop("tools", None) or [])
    for index, tool in enumerate(tools):
        # Lift Responses-shaped tools; leave Chat shapes to the public reader.
        # Flat functions retain permissive field selection, while hosted tools
        # prefer the provider's canonical name (for example Groq).
        if isinstance(tool, dict) and "function" not in tool:
            kind = tool.get("type")
            if kind == "function" and "name" in tool:
                if tool.get("strict") is True:
                    raise LMUnsupportedFeatureError(
                        "FunctionTool cannot represent strict=True; refusing to drop the requested tool enforcement."
                    )
                tool = FunctionTool(
                    name=tool["name"], description=tool.get("description"),
                    parameters=tool.get("parameters") or {"type": "object", "properties": {}},
                )
            elif isinstance(kind, str) and kind and kind != "function":
                try:
                    tool = request_from_openai_chat(
                        {"model": lm.model, "messages": _STUB_MESSAGES, "tools": [tool]}, compat=compat,
                    ).tools[0]
                except UnsupportedFeatureError:
                    tool = BuiltinTool(name=kind, config={key: value for key, value in tool.items() if key != "type"} or None)
        tools[index] = tool
    typed_tools = [tool for tool in tools if isinstance(tool, (FunctionTool, BuiltinTool))]
    body["tools"] = [tool for tool in tools if not isinstance(tool, (FunctionTool, BuiltinTool))]
    choice = body.pop("tool_choice", None)
    tool_choice = None
    if isinstance(choice, dict) and "function" not in choice and "allowed_tools" not in choice:
        kind = choice.get("type")
        if kind == "function" and isinstance(choice.get("name"), str):
            tool_choice = ToolChoice(mode="required", allowed=(choice["name"],))
        elif isinstance(kind, str) and kind in {tool.name for tool in typed_tools}:
            tool_choice = ToolChoice(mode="required", allowed=(kind,))
    format_ = body.get("response_format")
    if isinstance(format_, type) and issubclass(format_, pydantic.BaseModel):
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": format_.__name__, "schema": close_object_schemas(format_.model_json_schema()), "strict": True,
        }}
    try:
        # Name validation waits until typed and Chat tools are combined.
        if tool_choice is None:
            tool_choice = _openai_chat._ingest_tool_choice("openai-chat", choice, body.get("parallel_tool_calls"))
        read = request_from_openai_chat(body, compat=compat)
    except UnsupportedFeatureError as exc:
        raise LMUnsupportedFeatureError(str(exc), model=lm.model) from exc
    config = read.config
    if tool_choice is not None and body.get("parallel_tool_calls") is not None:
        tool_choice = replace(tool_choice, parallel=body["parallel_tool_calls"])
    if reasoning is not None and config.reasoning is not None and config.reasoning.is_off != reasoning.is_off:
        raise ValueError("reasoning and provider thinking controls are contradictory")
    reasoning = reasoning or config.reasoning
    if isinstance(native_thinking, dict) and reasoning is not None:
        mode = native_thinking.get("type")
        if mode in {"enabled", "adaptive", "disabled"} and reasoning.is_off != (mode == "disabled"):
            raise ValueError("reasoning and provider thinking controls are contradictory")
    if reasoning is not None and reasoning.summary is None and not reasoning.is_off:
        # DSPy reads native reasoning back into `dspy.Reasoning` fields, which
        # needs the provider to show a summary where one is available.
        reasoning = replace(reasoning, summary="auto")
    if prompt_cache is not None and config.cache is not None:
        raise LMUnsupportedFeatureError(
            "Do not combine prompt_cache with provider-shaped prompt-cache options.", model=lm.model,
        )
    try:
        config = replace(config, tool_choice=tool_choice, reasoning=reasoning,
                         cache=prompt_cache if prompt_cache is not None else config.cache,
                         extensions={**(config.extensions or {}), **passthrough} or None)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Provider options must be JSON values to travel in Config.extensions: {sorted(passthrough)}"
        ) from exc
    return Request(model=lm.model, messages=messages, system=system, tools=(*typed_tools, *read.tools), config=config)
