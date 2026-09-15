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
from dspy.lm15 import (
    BuiltinTool,
    CacheConfig,
    FunctionTool,
    Message,
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


def close_object_schemas(schema: Any) -> Any:
    """Strict structured outputs require every object schema to be closed."""
    if not isinstance(schema, dict):
        return schema
    closed = dict(schema)
    for key in ("items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"):
        if isinstance(closed.get(key), dict):
            closed[key] = close_object_schemas(closed[key])
    for key in ("items", "prefixItems", "anyOf", "oneOf", "allOf"):
        if isinstance(closed.get(key), list):
            closed[key] = [close_object_schemas(item) for item in closed[key]]
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        if isinstance(closed.get(key), dict):
            closed[key] = {name: close_object_schemas(sub) for name, sub in closed[key].items()}
    if closed.get("type") == "object" and "additionalProperties" not in closed:
        closed["additionalProperties"] = False
    return closed


def generation_options(lm, options: dict | None = None) -> dict[str, Any]:
    """LM defaults overridden by call options; None removes a default."""
    merged = {**lm.kwargs, **(options or {})}
    return {
        key: value for key, value in merged.items()
        if key not in EXECUTION_KEYS and key not in CLIENT_KEYS and value is not None
    }


def _typed_tool(tool):
    """Read the tool spellings DSPy accepts into lm15 tool objects.

    Chat Completions tools (`{"type": "function", "function": {...}}`) are left
    for lm15's reader. Responses API spellings are read here: a flat function
    tool becomes a FunctionTool; a hosted tool such as `{"type": "web_search"}`
    becomes a BuiltinTool carrying its other keys as config.
    """
    if isinstance(tool, (FunctionTool, BuiltinTool)):
        return tool
    if not isinstance(tool, dict) or "function" in tool:
        return None
    kind = tool.get("type")
    if kind == "function" and "name" in tool:
        return FunctionTool(
            name=tool["name"], description=tool.get("description"),
            parameters=tool.get("parameters") or {"type": "object", "properties": {}},
        )
    if isinstance(kind, str) and kind and kind != "function":
        config = {key: value for key, value in tool.items() if key != "type"}
        return BuiltinTool(name=kind, config=config or None)
    return None


def _typed_tool_choice(choice, tools):
    """Read a Responses API tool_choice into a ToolChoice, when it names a typed tool."""
    if not isinstance(choice, dict) or "function" in choice or "allowed_tools" in choice:
        return None
    kind = choice.get("type")
    names = {tool.name for tool in tools}
    if kind == "function" and choice.get("name") in names:
        return ToolChoice(mode="required", allowed=(choice["name"],))
    if isinstance(kind, str) and kind in names:
        return ToolChoice(mode="required", allowed=(kind,))
    return None


def _compat(lm, options):
    """The lm15 compat preset of the native route, when one will be used."""
    if not isinstance(lm.engine, str):
        return None
    try:
        selection = select_backend(lm, options)
    except Exception:
        return None
    return selection.resolution.compat if selection.native and selection.resolution is not None else None


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
    # The Responses API spells the reasoning dial as an object; DSPy has always
    # accepted that spelling on Responses models beside `reasoning_effort`.
    reasoning = body.pop("reasoning", None) if isinstance(body.get("reasoning"), dict) else None
    if reasoning is not None:
        body.pop("reasoning_effort", None)
    tools = list(body.pop("tools", None) or [])
    typed_tools = [typed for typed in map(_typed_tool, tools) if typed is not None]
    raw_tools = [tool for tool in tools if _typed_tool(tool) is None]
    if raw_tools:
        body["tools"] = raw_tools
    tool_choice = _typed_tool_choice(body.get("tool_choice"), typed_tools)
    if tool_choice is not None:
        body.pop("tool_choice")
    format_ = body.get("response_format")
    if isinstance(format_, type) and issubclass(format_, pydantic.BaseModel):
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": format_.__name__, "schema": close_object_schemas(format_.model_json_schema()), "strict": True,
        }}
    try:
        read = request_from_openai_chat(body, compat=_compat(lm, options))
    except UnsupportedFeatureError as exc:
        raise LMUnsupportedFeatureError(str(exc), model=lm.model) from exc
    config = read.config
    if tool_choice is not None:
        parallel = body.get("parallel_tool_calls")
        config = replace(config, tool_choice=replace(tool_choice, parallel=parallel) if parallel is not None else tool_choice)
    if reasoning is not None:
        config = replace(config, reasoning=Reasoning(effort=reasoning.get("effort", "medium"), summary=reasoning.get("summary")))
    if config.reasoning is not None and config.reasoning.summary is None and not config.reasoning.is_off:
        # DSPy reads native reasoning back into `dspy.Reasoning` fields, which
        # needs the provider to show a summary where one is available.
        config = replace(config, reasoning=replace(config.reasoning, summary="auto"))
    if prompt_cache is not None:
        if config.cache is not None:
            raise LMUnsupportedFeatureError(
                "Do not combine prompt_cache with provider-shaped prompt-cache options.", model=lm.model,
            )
        config = replace(config, cache=prompt_cache)
    if passthrough:
        try:
            config = replace(config, extensions={**(config.extensions or {}), **passthrough})
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Provider options must be JSON values to travel in Config.extensions: {sorted(passthrough)}"
            ) from exc
    return Request(model=lm.model, messages=messages, system=system, tools=(*typed_tools, *read.tools), config=config)
