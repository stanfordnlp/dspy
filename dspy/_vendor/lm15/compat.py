"""
lm15.compat — Typed provider/API compatibility policies.

Compatibility policies describe how a provider adapter should serialize a
canonical lm15 request for a specific API dialect. They are intentionally
separate from lm15.types: Request/Response describe *what* the caller wants;
compat profiles describe provider wire-format quirks.

Fields default to None, which means "inherit from the parent profile". The
string value "auto" is an explicit policy: ask the adapter to use its automatic
heuristic for that field. Keeping None distinct from "auto" matters because
profiles are layered.
"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field, fields
from typing import Literal, TypeAlias, get_args

from .types import JsonObject


# ─── Shared helpers ──────────────────────────────────────────────────


def _check_literal_or_none(value: object, literal_alias: object, field_name: str) -> None:
    if value is not None and value not in get_args(literal_alias):  # type: ignore[arg-type]
        raise ValueError(f"unsupported {field_name}: {value!r}")


def _check_json_object_or_none(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object or None")


def _merge_json_object(a: JsonObject | None, b: JsonObject | None) -> JsonObject | None:
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}


# ─── OpenAI Responses API compatibility ──────────────────────────────

OpenAIResponsesDeveloperRole = Literal["auto", "developer", "system"]
OpenAIResponsesMaxOutputTokensField = Literal[
    "auto",
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
]
OpenAIResponsesReasoningFormat = Literal[
    "auto",
    "none",
    "responses_reasoning",
    "reasoning_effort",
    "openrouter",
    "deepseek",
    "qwen",
    "qwen_chat_template",
    "zai",
]
OpenAIToolResultName = Literal["auto", "include", "omit"]
OpenAIStrictTools = Literal["auto", "include", "omit"]
# "openai": prompt_cache_key, prompt_cache_retention, and the explicit
# breakpoint mark (prompt_cache_breakpoint / prompt_cache_options, gpt-5.6+).
# "openai_implicit": the two documented fields only, on a server that caches
# every prefix automatically and would take a breakpoint mark without
# acting on it (Meta, guide--prompt-caching.md: "you do not … mark
# breakpoints"; live 2026-09-03 the mark answered 200 with no signal either
# way).  As with "none", an explicit CacheConfig is not an error there —
# the prefix is cached regardless — but no undocumented field is sent.
OpenAICacheControl = Literal["auto", "none", "openai", "openai_implicit", "anthropic"]
# Whether a replayed assistant message that precedes a function_call in the
# same turn carries `phase: "commentary"`.  Meta Model API stamps the field
# on the messages it emits before a tool call and documents replaying them
# untagged as HTTP 400 (protocols--responses.md § Message phase); live
# 2026-09-03 the untagged replay answered 200, so the tag is sent for the
# documented quality reason ("dropping it can degrade quality"), not to
# avoid an error.  OpenAI has no such field: "omit".
OpenAIResponsesCommentaryPhase = Literal["auto", "omit", "tag"]
# The multipart key for input images on POST /images/edits.  OpenAI takes
# `image[]` (pinned: cases/openai/image_edit.json); Meta rejects that key
# and asks for `image[0]`, `image[1]`, … (live 2026-09-03, HTTP 400
# "`image[]` is not a valid image key; use `image[N]`").
OpenAIResponsesEditImageField = Literal["auto", "array", "indexed"]
# Which server-executed tool types the canonical BuiltinTool names map to.
# "openai": OpenAI's Responses vocabulary (web_search_preview,
# code_interpreter, file_search, computer_use_preview).  "verbatim": the
# canonical name IS the wire type — servers whose schema spells the tool
# `web_search` (Meta responses--schemas.md; Moonshot responses--create.md)
# — and a name the server lacks goes out unchanged and is refused loudly
# (both: HTTP 400, live 2026-09-03).  Named for the shape, not the first
# server that needed it (renamed from "meta" on 2026-09-03, same day, before
# any port or release carried the old value).
OpenAIResponsesBuiltinTools = Literal["auto", "openai", "verbatim"]
# MAP-10 (2026-09-07): what a ToolResultPart's media parts may become on the
# wire.  "native" = images and documents as native blocks inside the result
# item; "images" = images only (documents raise); "reject" = every media part
# raises before the wire.  Measured per preset:
# lm15-contract/research/tool-result-content/30-model.md.
ToolResultMedia = Literal["auto", "native", "images", "reject"]


@dataclass(frozen=True, slots=True)
class OpenAIResponsesCompat:
    """Partial compatibility policy for OpenAI Responses-family APIs.

    None means "inherit". Non-None values override parent profiles. The value
    "auto" means "explicitly use adapter auto-detection".
    """

    developer_role: OpenAIResponsesDeveloperRole | None = None
    max_output_tokens_field: OpenAIResponsesMaxOutputTokensField | None = None
    reasoning_format: OpenAIResponsesReasoningFormat | None = None
    tool_result_name: OpenAIToolResultName | None = None
    strict_tools: OpenAIStrictTools | None = None
    cache_control: OpenAICacheControl | None = None
    # New knobs are keyword-only: routing/extensions keep their original slots.
    commentary_phase: OpenAIResponsesCommentaryPhase | None = field(default=None, kw_only=True)
    edit_image_field: OpenAIResponsesEditImageField | None = field(default=None, kw_only=True)
    builtin_tools: OpenAIResponsesBuiltinTools | None = field(default=None, kw_only=True)
    tool_result_media: ToolResultMedia | None = field(default=None, kw_only=True)
    routing: JsonObject | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_literal_or_none(self.tool_result_media, ToolResultMedia, "tool_result_media")
        _check_literal_or_none(self.developer_role, OpenAIResponsesDeveloperRole, "developer_role")
        _check_literal_or_none(self.commentary_phase, OpenAIResponsesCommentaryPhase, "commentary_phase")
        _check_literal_or_none(self.edit_image_field, OpenAIResponsesEditImageField, "edit_image_field")
        _check_literal_or_none(self.builtin_tools, OpenAIResponsesBuiltinTools, "builtin_tools")
        _check_literal_or_none(
            self.max_output_tokens_field,
            OpenAIResponsesMaxOutputTokensField,
            "max_output_tokens_field",
        )
        _check_literal_or_none(self.reasoning_format, OpenAIResponsesReasoningFormat, "reasoning_format")
        _check_literal_or_none(self.tool_result_name, OpenAIToolResultName, "tool_result_name")
        _check_literal_or_none(self.strict_tools, OpenAIStrictTools, "strict_tools")
        _check_literal_or_none(self.cache_control, OpenAICacheControl, "cache_control")
        _check_json_object_or_none(self.routing, "routing")
        _check_json_object_or_none(self.extensions, "extensions")

    @classmethod
    def preset(cls, name: str) -> "OpenAIResponsesCompat":
        """The named server dialect from :data:`OPENAI_RESPONSES_PRESETS`.

        Accepts the permanent spelling aliases (``responses``, ``lm-studio``,
        ``z.ai``); raises ``ValueError`` for an unknown name.
        """
        key = _preset_key(name)
        try:
            return OPENAI_RESPONSES_PRESETS[key]
        except KeyError:
            raise ValueError(f"unknown OpenAIResponsesCompat preset: {name!r}") from None


# Named Responses-dialect servers.  Same table shape as the chat dialect's
# OPENAI_CHAT_PRESETS (the if-chain it replaced was the fifth copy of
# provider knowledge the registry entry of 2026-09-03 set out to remove).
OPENAI_RESPONSES_PRESETS: dict[str, OpenAIResponsesCompat] = {
    "openai": OpenAIResponsesCompat(
        developer_role="developer",
        max_output_tokens_field="max_output_tokens",
        reasoning_format="responses_reasoning",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai",
    ),
    "openrouter": OpenAIResponsesCompat(
        developer_role="developer",
        max_output_tokens_field="max_tokens",
        reasoning_format="openrouter",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai",
        tool_result_media="reject",  # MAP-10: no receipt on this door — reject until one exists
    ),
    "ollama": OpenAIResponsesCompat(
        developer_role="system",
        max_output_tokens_field="max_tokens",
        reasoning_format="none",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: no receipt on this door — reject until one exists
    ),
    "vllm": OpenAIResponsesCompat(
        developer_role="system",
        max_output_tokens_field="max_tokens",
        reasoning_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: no receipt on this door — reject until one exists
    ),
    "sglang": OpenAIResponsesCompat(
        developer_role="system",
        max_output_tokens_field="max_tokens",
        reasoning_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: no receipt on this door — reject until one exists
    ),
    "qwen": OpenAIResponsesCompat(
        developer_role="system",
        max_output_tokens_field="max_tokens",
        reasoning_format="qwen",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: no receipt on this door — reject until one exists
    ),
    "deepseek": OpenAIResponsesCompat(
        developer_role="system",
        max_output_tokens_field="max_tokens",
        reasoning_format="deepseek",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: its Chat and Messages doors both show the model [Unsupported Image] — reject
    ),
    "zai": OpenAIResponsesCompat(
        developer_role="system",
        max_output_tokens_field="max_tokens",
        reasoning_format="zai",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: no receipt on this door (zai Chat carries images) — reject until one exists
    ),
    # Meta Model API (dev.meta.ai, scraped 2026-09-03 —
    # lm15-contract/scrapes/meta/pages): the Responses wire as OpenAI
    # speaks it.  `developer` role (protocols--responses.md), `max_output_
    # tokens` (min 16), `reasoning.effort` + `reasoning.summary`
    # (guide--reasoning.md; `none` is refused loudly by Muse Spark),
    # `prompt_cache_key` and `prompt_cache_retention` honoured (guide--
    # prompt-caching.md); the breakpoint mark is never sent (cache_control
    # "openai_implicit": the server accepts it silently, live 2026-09-03).
    "meta": OpenAIResponsesCompat(
        developer_role="developer",
        max_output_tokens_field="max_output_tokens",
        reasoning_format="responses_reasoning",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai_implicit",
        commentary_phase="tag",
        edit_image_field="indexed",
        builtin_tools="verbatim",
        tool_result_media="native",  # MAP-10: FunctionCallOutputContentListParam; image/mixed/pair/pdf received (muse-spark-1.3)
    ),
    # Moonshot AI over the Responses wire (platform.kimi.ai responses--
    # create.md OpenAPI, scraped 2026-09-03): kimi-k3 only; `instructions`
    # + `developer` role; `max_output_tokens`; `reasoning.effort` low|high|
    # max; stateless (`store` always false, `previous_response_id` always
    # null) and reasoning comes back as `summary` text with
    # `encrypted_content: null`, replayed as-is; `prompt_cache_key` and
    # `safety_identifier`; tools function|custom|namespace|web_search — the
    # canonical name is the wire type (builtin_tools "verbatim").
    "moonshotai": OpenAIResponsesCompat(
        developer_role="developer",
        max_output_tokens_field="max_output_tokens",
        reasoning_format="responses_reasoning",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai_implicit",
        builtin_tools="verbatim",
        tool_result_media="images",  # MAP-10: images received (kimi-k3); input_file is 400 "unknown content part type"
    ),
}

# Default base URLs for the Responses presets that name a server (the
# registry's access policies point here so each URL has one copy).
OPENAI_RESPONSES_PRESET_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    # dev.meta.ai overview.md: one base URL for every Meta Model API surface.
    "meta": "https://api.meta.ai/v1",
    # platform.kimi.ai api--overview.md: /v1/responses on the OpenAI root.
    "moonshotai": "https://api.moonshot.ai/v1",
}


@dataclass(frozen=True, slots=True)
class ResolvedOpenAIResponsesCompat:
    """Fully resolved OpenAI Responses compatibility policy."""

    developer_role: Literal["developer", "system"] = "developer"
    max_output_tokens_field: Literal["max_output_tokens", "max_completion_tokens", "max_tokens"] = "max_output_tokens"
    reasoning_format: Literal[
        "none",
        "responses_reasoning",
        "reasoning_effort",
        "openrouter",
        "deepseek",
        "qwen",
        "qwen_chat_template",
        "zai",
    ] = "responses_reasoning"
    tool_result_name: Literal["include", "omit"] = "omit"
    strict_tools: Literal["include", "omit"] = "omit"
    cache_control: Literal["none", "openai", "openai_implicit", "anthropic"] = "openai"
    commentary_phase: Literal["omit", "tag"] = field(default="omit", kw_only=True)
    edit_image_field: Literal["array", "indexed"] = field(default="array", kw_only=True)
    builtin_tools: Literal["openai", "verbatim"] = field(default="openai", kw_only=True)
    # The documented Responses wire takes input_image and input_file in a
    # function_call_output array (MAP-10; openai gpt-5.4 exact, 2026-09-07).
    tool_result_media: Literal["native", "images", "reject"] = field(default="native", kw_only=True)
    routing: JsonObject | None = None
    extensions: JsonObject | None = None


# ─── OpenAI Chat Completions compatibility ───────────────────────────

# Knobs a per-model override may set (OpenAIChatCompat.model_overrides).
_CHAT_OVERRIDABLE: frozenset[str] = frozenset({
    "instruction_role", "max_tokens_field", "stream_usage", "thinking_format", "thinking_replay",
    "assistant_reasoning_content", "strict_tools", "cache_control", "user_field",
    "forced_tool_choice", "json_schema", "reasoning_efforts", "tool_result_media",
})

OpenAIChatInstructionRole = Literal["auto", "developer", "system"]
OpenAIChatMaxTokensField = Literal["auto", "max_completion_tokens", "max_tokens"]
OpenAIChatStreamUsage = Literal["auto", "include", "omit"]
OpenAIChatAssistantAfterToolResult = Literal["auto", "insert", "omit"]
OpenAIChatThinkingReplay = Literal["auto", "native", "as_text", "omit"]
OpenAIChatAssistantReasoningContent = Literal["auto", "include_empty", "omit"]
# "deepseek" names a wire SHAPE — thinking={"type": enabled|disabled} plus
# reasoning_effort — not a company: DeepSeek, xAI and Z.AI all speak it
# (docs.z.ai chat--create.md ChatThinking; the earlier "zai" value sent
# Qwen's enable_thinking and was never live-validated; removed 2026-09-03).
# "kimi" is the union of Moonshot's two documented shapes, split by intent
# rather than by model: an effort word goes out as top-level
# `reasoning_effort` alone (kimi-k3: low|high|max, api--models-overview.md),
# and off goes out as `thinking: {type: disabled}` alone (kimi-k2.6).  Each
# family rejects the other family's field loudly (live 2026-09-03), so
# neither is ever sent alongside the one the docs name for that intent.
OpenAIChatThinkingFormat = Literal[
    "auto",
    "none",
    "reasoning_effort",
    "openrouter",
    "deepseek",
    "kimi",
    "qwen",
    "qwen_chat_template",
]
# BuiltinTool policy for the chat dialect. The base Chat Completions wire
# carries function/custom tools ONLY (doc: chat--create.md), and some
# compat servers silently IGNORE unknown tool types (OpenRouter, verified
# live 2026-09-01) — so "reject" (raise) is the only safe default.
# "groq" maps canonical builtin names onto Groq's server-executed tool
# types (browser_search / code_interpreter, both verified live
# 2026-09-01).
OpenAIChatBuiltinTools = Literal["auto", "reject", "groq"]
# Which request field carries Config.user_id.  OpenAI's dialect spells it
# `user`; DeepSeek documents `user_id` (content-safety, KV-cache and
# scheduling isolation, rate-limit.md) and accepts `user` silently (live
# 2026-09-03: 200 either way, no echo) — so the documented name is the
# only one that can be trusted to do anything.  Meta deprecates `user` in
# favour of `safety_identifier` (protocols--chat-completions.md § Parameters),
# the same field the Responses dialect already sends.
OpenAIChatUserField = Literal["auto", "user", "user_id", "safety_identifier"]
# Whether the server honours a tool_choice other than "auto" (required,
# none, a named function, an allowlist).  "reject" makes the adapter raise
# UnsupportedFeatureError before the wire.  Z.AI documents auto only and
# ignores the rest silently (live 2026-09-03: mode=required answered text,
# mode=none called the tool) — a silent widen is worse than an error (MAP-8).
OpenAIChatForcedToolChoice = Literal["auto", "send", "reject"]
# Whether the server honours response_format.type=json_schema.  "reject"
# raises before the wire.  DeepSeek answers 400 (loud, nothing to do); Z.AI
# answers 200 with free-form, fenced JSON that ignores the schema (live
# 2026-09-03) — silent, so the adapter must refuse.
OpenAIChatJsonSchema = Literal["auto", "send", "reject"]
# The server's native reasoning-effort levels, when the server does NOT
# refuse the others.  MAP-7 rule 2: a word with no native level raises
# client-side rather than downgrading silently.  Most servers answer 400 to
# an unknown word and need nothing here (None); Moonshot's kimi-k3 documents
# low|high|max and answers 200 to `medium` and to `bogus` alike (live
# 2026-09-03), so the preset lists the three.  `off` is never in the list:
# the off switch is thinking_format's business.
OpenAIChatReasoningEfforts = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpenAIChatCompat:
    """Partial compatibility policy for OpenAI Chat Completions-family APIs.

    This class is consumed by OpenAIChatLM (lm15.providers.openai_chat); it is
    kept separate so profiles can describe chat-completions style endpoints
    without overloading OpenAIResponsesCompat.
    """

    instruction_role: OpenAIChatInstructionRole | None = None
    max_tokens_field: OpenAIChatMaxTokensField | None = None
    stream_usage: OpenAIChatStreamUsage | None = None
    tool_result_name: OpenAIToolResultName | None = None
    assistant_after_tool_result: OpenAIChatAssistantAfterToolResult | None = None
    thinking_format: OpenAIChatThinkingFormat | None = None
    thinking_replay: OpenAIChatThinkingReplay | None = None
    assistant_reasoning_content: OpenAIChatAssistantReasoningContent | None = None
    strict_tools: OpenAIStrictTools | None = None
    builtin_tools: OpenAIChatBuiltinTools | None = None
    tool_result_media: ToolResultMedia | None = field(default=None, kw_only=True)
    cache_control: OpenAICacheControl | None = None
    user_field: OpenAIChatUserField | None = None
    forced_tool_choice: OpenAIChatForcedToolChoice | None = None
    json_schema: OpenAIChatJsonSchema | None = None
    reasoning_efforts: OpenAIChatReasoningEfforts | None = field(default=None, kw_only=True)
    routing: JsonObject | None = None
    extensions: JsonObject | None = None
    # Per-model-family overrides on a door that forwards knobs to many
    # vendors' models (Bedrock's Chat Completions door, live 2026-09-03:
    # DeepSeek/Mistral/Qwen/Z.AI honour tool_choice=required and
    # json_schema, gpt-oss ignores both, Gemma ignores tool_choice).  Each
    # entry is (model-id prefix, {knob: value}); the first matching prefix
    # wins; the knobs are the fields above.  Resolved per request.
    model_overrides: tuple[tuple[str, dict[str, str]], ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        overrides = []
        for prefix, knobs in self.model_overrides:
            if not isinstance(prefix, str) or not prefix:
                raise ValueError("model_overrides: each prefix is a non-empty string")
            knobs = dict(knobs)
            for name in knobs:
                if name not in _CHAT_OVERRIDABLE:
                    raise ValueError(f"model_overrides: {name!r} is not an overridable knob ({sorted(_CHAT_OVERRIDABLE)})")
            overrides.append((prefix, knobs))
        object.__setattr__(self, "model_overrides", tuple(overrides))
        _check_literal_or_none(self.instruction_role, OpenAIChatInstructionRole, "instruction_role")
        _check_literal_or_none(self.max_tokens_field, OpenAIChatMaxTokensField, "max_tokens_field")
        _check_literal_or_none(self.stream_usage, OpenAIChatStreamUsage, "stream_usage")
        _check_literal_or_none(self.tool_result_name, OpenAIToolResultName, "tool_result_name")
        _check_literal_or_none(
            self.assistant_after_tool_result,
            OpenAIChatAssistantAfterToolResult,
            "assistant_after_tool_result",
        )
        _check_literal_or_none(self.thinking_format, OpenAIChatThinkingFormat, "thinking_format")
        _check_literal_or_none(self.thinking_replay, OpenAIChatThinkingReplay, "thinking_replay")
        _check_literal_or_none(
            self.assistant_reasoning_content,
            OpenAIChatAssistantReasoningContent,
            "assistant_reasoning_content",
        )
        _check_literal_or_none(self.strict_tools, OpenAIStrictTools, "strict_tools")
        _check_literal_or_none(self.builtin_tools, OpenAIChatBuiltinTools, "builtin_tools")
        _check_literal_or_none(self.tool_result_media, ToolResultMedia, "tool_result_media")
        _check_literal_or_none(self.cache_control, OpenAICacheControl, "cache_control")
        _check_literal_or_none(self.user_field, OpenAIChatUserField, "user_field")
        _check_literal_or_none(self.forced_tool_choice, OpenAIChatForcedToolChoice, "forced_tool_choice")
        _check_literal_or_none(self.json_schema, OpenAIChatJsonSchema, "json_schema")
        if self.reasoning_efforts is not None:
            from .types import REASONING_EFFORTS

            bad = [w for w in self.reasoning_efforts if w not in REASONING_EFFORTS or w == "off"]
            if not isinstance(self.reasoning_efforts, tuple) or bad:
                raise ValueError(
                    f"reasoning_efforts must be a tuple of ReasoningEffort words other than 'off'; got {self.reasoning_efforts!r}"
                )
        _check_json_object_or_none(self.routing, "routing")
        _check_json_object_or_none(self.extensions, "extensions")


    def for_model(self, model: str) -> "OpenAIChatCompat":
        """This compat with the first matching ``model_overrides`` entry applied."""
        for prefix, knobs in self.model_overrides:
            if model.startswith(prefix):
                return dataclasses.replace(self, model_overrides=(), **knobs)
        return self

    @classmethod
    def preset(cls, name: str) -> "OpenAIChatCompat":
        """The named server dialect from :data:`OPENAI_CHAT_PRESETS`.

        Accepts the permanent spelling aliases (``lm-studio``, ``z.ai``,
        ``openai_chat``); raises ``ValueError`` for an unknown name.
        """
        key = _preset_key(name)
        try:
            return OPENAI_CHAT_PRESETS[key]
        except KeyError:
            raise ValueError(f"unknown OpenAIChatCompat preset: {name!r}") from None


def _preset_key(name: str) -> str:
    key = name.lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    return _OPENAI_CHAT_PRESET_ALIASES.get(key, key)


# Spelling aliases → canonical preset key.  Every alias is permanent.  One
# map serves all three dialect tables: a name means the same server in each.
_OPENAI_CHAT_PRESET_ALIASES: dict[str, str] = {
    "openai_chat": "openai",
    "chat": "openai",
    "chat_completions": "openai",
    "responses": "openai",
    "openai_responses": "openai",
    "lmstudio": "ollama",
    "lm_studio": "ollama",
    "dashscope_qwen": "qwen",
    "z_ai": "zai",
}

# The Chat Completions server dialects lm15 knows, as data.  A preset
# describes ONE server's quirks (instruction role, token-limit field,
# thinking wire shape, …); it says nothing about credentials or routing —
# that is the provider registry (lm15.registry), which names a preset by
# key.  Every field value here is pinned by a live receipt or the server's
# own documentation, cited inline; a preset changes with new evidence, not
# with a hunch.
OPENAI_CHAT_PRESETS: dict[str, OpenAIChatCompat] = {
    "openai": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_completion_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai",
        tool_result_media="reject",  # MAP-10: text-only tool row; gpt-5.4 received the USER image and not the tool image
    ),
    # ollama / LM Studio: max_tokens, no reasoning dial on the wire.
    "ollama": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="none",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: openai.go builds image rows without ToolCallID; no receipt — reject until one exists
    ),
    # Groq: server-executed builtin tools (browser_search / code_interpreter,
    # live 2026-09-01); reasoning_effort dial; no cache_control field.
    "groq": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        builtin_tools="groq",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: server 400 "messages[2].content must be a string" despite the SDK union
    ),
    # OpenRouter: unified reasoning object; OpenAI-shaped cache_control.
    "openrouter": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="openrouter",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai",
        tool_result_media="reject",  # MAP-10: schema says image; no receipt (401 during the pass) — reject until one exists
    ),
    # xAI, pinned live 2026-09-01 against grok-4.6: max_tokens accepted,
    # reasoning arrives as message.reasoning_content (deepseek shape),
    # stream_options.include_usage honored, no cache_control field (prompt
    # caching is automatic; usage reports cached_tokens).
    "xai": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="deepseek",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="images",  # MAP-10: image tool results received (grok-4.20); documents 400 "use /v1/responses"
    ),
    "vllm": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: parser carries it; no server reachable in the pass — reject until a receipt
    ),
    "sglang": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        tool_result_media="reject",  # MAP-10: schema carries it; no server reachable in the pass — reject until a receipt
    ),
    # DeepSeek (api-docs.deepseek.com, scraped 2026-09-03 —
    # lm15-contract/scrapes/deepseek/pages): thinking={"type": enabled|
    # disabled} + reasoning_effort (guide--thinking-mode.md); max_tokens
    # (chat--create.md); usage in the final stream chunk via
    # stream_options.include_usage; context caching is automatic on disk,
    # no cache_control field (guide--kv-cache.md).  thinking_replay=native
    # + include_empty: with tools present, every assistant turn must carry
    # reasoning_content back — even turns that made no call — or the API
    # answers 400 (guide--thinking-mode.md § Tool Calls).
    "deepseek": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="deepseek",
        thinking_replay="native",
        assistant_reasoning_content="include_empty",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        user_field="user_id",
        tool_result_media="reject",  # MAP-10: HTTP 200 and the model sees [Unsupported Image] — silent degrade
    ),
    "qwen": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="qwen",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
    ),
    # Z.AI (docs.z.ai, scraped 2026-09-03 — lm15-contract/scrapes/zai/pages):
    # thinking={"type": enabled|disabled} + reasoning_effort (chat--create.md
    # ChatThinking, the deepseek wire shape; GLM-5.3 cannot disable and the
    # request fails loudly — model--glm-5.3.md); max_tokens; interleaved
    # thinking asks for reasoning_content to be returned with tool results
    # (guide--thinking-mode.md); implicit context caching, usage in
    # prompt_tokens_details.cached_tokens (guide--cache.md); user_id 6–128
    # chars is the documented identity field (chat--create.md).
    # Amazon Bedrock's OpenAI Chat Completions door on bedrock-runtime
    # (live 2026-09-03, openai.gpt-oss-20b-1:0, SigV4; receipts/2026-09-03-
    # bedrock-chat/): system role, max_completion_tokens, usage on the final
    # chunk, reasoning_effort accepted, `user` accepted.  The door forwards
    # tool_choice and response_format to every vendor's model; whether a
    # model honours them is per family (six families probed, family-*.json):
    # the two that ignore them are refused per model (MAP-8) below.
    "bedrock": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_completion_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        user_field="user",
        forced_tool_choice="send",
        json_schema="send",
        model_overrides=(
            # receipts/2026-09-03-bedrock-chat/family-*.json: HTTP 200, the
            # knob sent, the model answered as if it were absent.
            ("openai.gpt-oss", {"forced_tool_choice": "reject", "json_schema": "reject"}),
            ("google.gemma", {"forced_tool_choice": "reject"}),
        ),
        tool_result_media="reject",  # MAP-10: 400 validation_error on the array form
    ),
    # Bedrock-mantle Chat Completions (live 2026-09-04, openai.gpt-oss-20b and
    # deepseek.v3.2; receipts/2026-09-04-bedrock-chat/probe-mantle-chat-* and
    # receipts/<date>-bedrock-mantle-chat/).  Same Chat Completions dialect as
    # `bedrock`, a different host: un-versioned ids, a working /models list,
    # gpt-oss reasoning in `message.reasoning` (the parser already reads that
    # field).  tool_choice / json_schema overrides land from the family
    # probes of this door, not copied from `bedrock` — mantle translates,
    # so a silent ignore on runtime is not evidence here.  Live 2026-09-04
    # family probes: DeepSeek/Mistral/Qwen/Z.AI/Gemma honour both knobs
    # (Gemma honours tool_choice here; it ignores it on bedrock-runtime);
    # gpt-oss ignores both (HTTP 200, plain text / prose+JSON).
    "bedrock_mantle": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_completion_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        user_field="user",
        forced_tool_choice="send",
        json_schema="send",
        model_overrides=(
            ("openai.gpt-oss", {"forced_tool_choice": "reject", "json_schema": "reject"}),
        ),
    ),
    "zai": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="deepseek",
        thinking_replay="native",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        user_field="user_id",
        forced_tool_choice="reject",
        json_schema="reject",
        tool_result_media="images",  # MAP-10: image tool results received (glm-4.6v); a file part is 400 (code 1210)
    ),
    # Meta Model API (dev.meta.ai, scraped 2026-09-03 — lm15-contract/scrapes/
    # meta/pages, protocols--chat-completions.md): `developer` is the
    # documented instruction role (`system` is merged at the same level);
    # `max_completion_tokens` (`max_tokens` is a deprecated alias); top-level
    # `reasoning_effort` (Muse Spark always reasons — `none` is HTTP 400,
    # loud); `stream_options.include_usage` documented (chat--schemas.md);
    # `prompt_cache_key` honoured, caching otherwise automatic (guide--
    # prompt-caching.md); `safety_identifier` supersedes `user`.  The
    # `reasoning_content` field on this wire is redacted to empty for
    # external keys (guide--reasoning.md § Chat Completions), so nothing is
    # replayed: thinking_replay stays at the dialect default.
    "meta": OpenAIChatCompat(
        instruction_role="developer",
        max_tokens_field="max_completion_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai_implicit",
        user_field="safety_identifier",
        tool_result_media="reject",  # MAP-10: 400 "did not match any supported type"; Meta's Responses door carries media
    ),
    # Moonshot AI / Kimi API Platform (platform.kimi.ai, scraped 2026-09-03 —
    # lm15-contract/scrapes/moonshotai/pages, chat--create.md OpenAPI):
    # `role: system` instructions; `max_completion_tokens` (`max_tokens` is
    # marked deprecated); `stream_options.include_usage` documented; the
    # reasoning dial is split by model family — kimi-k3 takes top-level
    # `reasoning_effort` (low|high|max, always thinks), kimi-k2.6 takes
    # `thinking: {type}` (guide--thinking-models.md) — which the "kimi" shape
    # expresses by intent; every model returns `reasoning_content` and the
    # docs require it back verbatim in multi-turn and tool loops
    # (thinking_replay=native); context caching is automatic with
    # `prompt_cache_key` as the only knob (guide--context-caching.md);
    # `safety_identifier` is the documented identity field (no `user`).
    "moonshotai": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_completion_tokens",
        stream_usage="include",
        thinking_format="kimi",
        thinking_replay="native",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai_implicit",
        user_field="safety_identifier",
        reasoning_efforts=("low", "high", "max"),
        tool_result_media="images",  # MAP-10: image tool results received (kimi-k2.6); a file part is 400
    ),
}


# Default base URLs for the Chat Completions presets that name a server.
# Used by OpenAIChatLM when a compat preset is given by name and no
# explicit base_url overrides it; the provider registry's access policies
# point here so there is one copy of each URL.
OPENAI_CHAT_PRESET_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "xai": "https://api.x.ai/v1",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    # api-docs.deepseek.com/first-call.md: base_url https://api.deepseek.com
    # (no /v1; the site also answers /v1, but the documented form wins).
    "deepseek": "https://api.deepseek.com",
    # docs.z.ai introduction.md: the general endpoint.  The GLM Coding Plan
    # uses a different, subscription-bound endpoint that lm15 does not name.
    "zai": "https://api.z.ai/api/paas/v4",
    # dev.meta.ai overview.md: one base URL for every Meta Model API surface.
    "meta": "https://api.meta.ai/v1",
    # platform.kimi.ai api--overview.md: the OpenAI-compatible root (Chat
    # Completions and Responses); the Anthropic wire lives at /anthropic.
    "moonshotai": "https://api.moonshot.ai/v1",
}


@dataclass(frozen=True, slots=True)
class ResolvedOpenAIChatCompat:
    """Fully resolved OpenAI Chat Completions compatibility policy."""

    instruction_role: Literal["developer", "system"] = "system"
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    stream_usage: Literal["include", "omit"] = "include"
    tool_result_name: Literal["include", "omit"] = "omit"
    assistant_after_tool_result: Literal["insert", "omit"] = "omit"
    thinking_format: Literal[
        "none",
        "reasoning_effort",
        "openrouter",
        "deepseek",
        "kimi",
        "qwen",
        "qwen_chat_template",
    ] = "reasoning_effort"
    thinking_replay: Literal["native", "as_text", "omit"] = "as_text"
    assistant_reasoning_content: Literal["include_empty", "omit"] = "omit"
    strict_tools: Literal["include", "omit"] = "omit"
    builtin_tools: Literal["reject", "groq"] = "reject"
    # The base Chat wire's tool row takes text only (OpenAI's own schema; a
    # 200 with the image not received on gpt-5.4, 2026-09-07): reject unless
    # a preset proved the array form live (MAP-10).
    tool_result_media: Literal["native", "images", "reject"] = field(default="reject", kw_only=True)
    cache_control: Literal["none", "openai", "openai_implicit", "anthropic"] = "openai"
    user_field: Literal["user", "user_id"] = "user"
    forced_tool_choice: Literal["send", "reject"] = "send"
    json_schema: Literal["send", "reject"] = "send"
    reasoning_efforts: tuple[str, ...] | None = field(default=None, kw_only=True)
    routing: JsonObject | None = None
    extensions: JsonObject | None = None


_CHAT_AUTO_DEFAULTS: dict[str, str] = {
    "instruction_role": "system",
    "max_tokens_field": "max_completion_tokens",
    "stream_usage": "include",
    "tool_result_name": "omit",
    "assistant_after_tool_result": "omit",
    "thinking_format": "reasoning_effort",
    "thinking_replay": "as_text",  # decision G (2026-09-01): unsigned thinking is replayed as text, never dropped
    "assistant_reasoning_content": "omit",
    "strict_tools": "omit",
    "builtin_tools": "reject",
    "tool_result_media": "reject",
    "cache_control": "openai",
    "user_field": "user",
    "forced_tool_choice": "send",
    "json_schema": "send",
}


def merge_openai_chat_compat(
    base: OpenAIChatCompat,
    override: OpenAIChatCompat | None,
) -> OpenAIChatCompat:
    """Merge partial OpenAI Chat compat objects.

    None fields inherit. Non-None fields, including "auto", override.
    """
    if override is None:
        return base

    kwargs = {}
    for f in fields(OpenAIChatCompat):
        value = getattr(override, f.name)
        if f.name == "extensions":
            kwargs[f.name] = _merge_json_object(base.extensions, override.extensions)
        else:
            kwargs[f.name] = getattr(base, f.name) if value is None else value
    return OpenAIChatCompat(**kwargs)


def resolve_openai_chat_compat(partial: OpenAIChatCompat) -> ResolvedOpenAIChatCompat:
    """Resolve a partial chat compat object into concrete serializer policy."""
    kwargs: dict[str, object] = {}
    for field_name, default in _CHAT_AUTO_DEFAULTS.items():
        value = getattr(partial, field_name)
        kwargs[field_name] = default if value in {None, "auto"} else value
    return ResolvedOpenAIChatCompat(
        reasoning_efforts=partial.reasoning_efforts,
        routing=partial.routing,
        extensions=partial.extensions,
        **kwargs,  # type: ignore[arg-type]
    )


# ─── Anthropic Messages compatibility ───────────────────────────────
#
# The Messages wire has few dialects, so this policy is small.  It exists
# for servers that speak the format but not all of it: DeepSeek's Anthropic
# endpoint (api-docs.deepseek.com guide--anthropic-api.md, live 2026-09-03)
# is the first.  Fields default to None (inherit), "auto" is the dialect's
# own behavior, same convention as the OpenAI policies above.

# How reasoning is switched and steered.  "anthropic": the model-class rule
# (MAP-7 rule 10: manual class takes thinking.type=enabled + budget_tokens,
# adaptive class takes thinking.type=adaptive + output_config.effort; off
# is absence).  "deepseek": thinking.type=enabled|disabled +
# output_config.effort, no budget (the server ignores budget_tokens), and
# off MUST be sent — absence means on for DeepSeek.  "adaptive": every
# model is the adaptive class (thinking.type=adaptive + output_config.
# effort; `enabled` + budget_tokens is accepted but not translated) and
# off MUST be sent as thinking.type=disabled — the server refuses it with
# HTTP 400 rather than silently reasoning anyway (Meta Model API,
# protocols--messages.md § Reasoning).  "effort": the server documents no
# `thinking` request field at all — output_config.effort is the whole dial
# and goes out alone; off is sent as thinking.type=disabled, which the
# server honours (Moonshot kimi-k3, messages--create.md; live 2026-09-03).
AnthropicThinkingFormat = Literal["auto", "anthropic", "deepseek", "adaptive", "effort"]
# How a ThinkingPart that carries no signature is replayed.  "signed": the
# dialect's rule — only a signed block goes back as `thinking`; an unsigned
# one goes back as text (decision G: api.anthropic.com rejects unsigned
# thinking blocks).  "unsigned": the block goes back as `thinking` without
# a signature — for servers that return `signature: ""` and accept the
# block back (Moonshot, live 2026-09-03), where a text replay would put
# the model's reasoning into its spoken turn.
AnthropicThinkingReplay = Literal["auto", "signed", "unsigned"]
# Whether temperature / top_p / top_k reach the wire.  "reject" raises
# before the wire on a server that documents none of them and swallows
# them silently (Moonshot: `temperature: 0.5` is HTTP 200 here while the
# same server's chat wire answers "only 1 is allowed for this model" —
# live 2026-09-03; MAP-8 §2).
AnthropicSamplingParams = Literal["auto", "send", "reject"]
# Whether cache_control marks are placed.  "none": the server ignores marks
# and caches implicitly; nothing is placed and, as on the chat dialect's
# "none", an explicit CacheConfig is not an error (implicit caching applies).
AnthropicCacheControl = Literal["auto", "anthropic", "none"]
# Whether output_config.format (a JSON schema) is honoured.  "reject" raises
# before the wire: DeepSeek answers 200 and ignores the schema (live
# 2026-09-03) — silent, so the adapter refuses.
AnthropicStructuredOutput = Literal["auto", "send", "reject"]
# Whether disable_parallel_tool_use is honoured.  DeepSeek documents it as
# ignored; a silent no-op on ToolChoice.parallel=False is refused (MAP-8 §2).
AnthropicParallelToolCalls = Literal["auto", "send", "reject"]


@dataclass(frozen=True, slots=True)
class AnthropicCompat:
    """Partial compatibility policy for Anthropic Messages-family APIs.

    ``model_prefixes`` refuses, before the wire, any model id that does not
    start with one of the prefixes: DeepSeek's endpoint silently serves
    ``claude-opus*`` as ``deepseek-v4-pro`` and ``claude-haiku*`` /
    ``claude-sonnet*`` as ``deepseek-v4-flash`` (docs and live 2026-09-03);
    lm15 does not take part in a substitution the caller cannot see.
    None means any model id goes out as typed.
    """

    thinking_format: AnthropicThinkingFormat | None = None
    # Preserve the original thinking_format/cache_control/... positional API.
    thinking_replay: AnthropicThinkingReplay | None = field(default=None, kw_only=True)
    cache_control: AnthropicCacheControl | None = None
    structured_output: AnthropicStructuredOutput | None = None
    parallel_tool_calls: AnthropicParallelToolCalls | None = None
    sampling_params: AnthropicSamplingParams | None = field(default=None, kw_only=True)
    tool_result_media: ToolResultMedia | None = field(default=None, kw_only=True)
    reasoning_efforts: tuple[str, ...] | None = field(default=None, kw_only=True)
    model_prefixes: tuple[str, ...] | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_literal_or_none(self.tool_result_media, ToolResultMedia, "tool_result_media")
        _check_literal_or_none(self.thinking_format, AnthropicThinkingFormat, "thinking_format")
        _check_literal_or_none(self.thinking_replay, AnthropicThinkingReplay, "thinking_replay")
        _check_literal_or_none(self.cache_control, AnthropicCacheControl, "cache_control")
        _check_literal_or_none(self.structured_output, AnthropicStructuredOutput, "structured_output")
        _check_literal_or_none(self.parallel_tool_calls, AnthropicParallelToolCalls, "parallel_tool_calls")
        _check_literal_or_none(self.sampling_params, AnthropicSamplingParams, "sampling_params")
        if self.reasoning_efforts is not None:
            from .types import REASONING_EFFORTS

            bad = [w for w in self.reasoning_efforts if w not in REASONING_EFFORTS or w == "off"]
            if not isinstance(self.reasoning_efforts, tuple) or bad:
                raise ValueError(
                    f"reasoning_efforts must be a tuple of ReasoningEffort words other than 'off'; got {self.reasoning_efforts!r}"
                )
        if self.model_prefixes is not None:
            object.__setattr__(self, "model_prefixes", tuple(str(p) for p in self.model_prefixes))
            if not self.model_prefixes:
                raise ValueError("model_prefixes must be None or a non-empty tuple")
        _check_json_object_or_none(self.extensions, "extensions")

    @classmethod
    def preset(cls, name: str) -> "AnthropicCompat":
        key = _preset_key(name)
        try:
            return ANTHROPIC_PRESETS[key]
        except KeyError:
            raise ValueError(f"unknown AnthropicCompat preset: {name!r}") from None


ANTHROPIC_PRESETS: dict[str, AnthropicCompat] = {
    "anthropic": AnthropicCompat(),
    # DeepSeek over the Anthropic wire (api-docs.deepseek.com
    # guide--anthropic-api.md; live 2026-09-03, receipts/2026-09-03-deepseek-anthropic):
    # thinking={"type": enabled|disabled} + output_config.effort, budget_tokens
    # ignored; cache_control ignored (implicit caching reports
    # cache_read_input_tokens); output_config.format accepted and ignored;
    # disable_parallel_tool_use ignored; claude-* model names silently
    # remapped to DeepSeek models.
    "deepseek": AnthropicCompat(
        thinking_format="deepseek",
        cache_control="none",
        structured_output="reject",
        parallel_tool_calls="reject",
        model_prefixes=("deepseek-",),
        tool_result_media="reject",  # MAP-10: 200 and the model sees [Unsupported Image] — silent degrade
    ),
    # Meta Model API over the Anthropic wire (dev.meta.ai protocols--
    # messages.md, scraped 2026-09-03): thinking={"type": "adaptive"} +
    # output_config.effort low|medium|high|xhigh, `disabled` is HTTP 400;
    # caching is automatic (guide--prompt-caching.md names no cache_control
    # block on this wire) so no marks are placed; output_config.format
    # (json_schema) and disable_parallel_tool_use are documented as honoured.
    "meta": AnthropicCompat(
        thinking_format="adaptive",
        cache_control="none",
        structured_output="send",
        parallel_tool_calls="send",
        tool_result_media="native",  # MAP-10: image/mixed/pair/pdf received (muse-spark-1.3)
    ),
    # Moonshot AI over the Anthropic wire (platform.kimi.ai messages--
    # create.md OpenAPI, scraped 2026-09-03): kimi-k3 only, always reasons;
    # no `thinking` request field at all — `output_config.effort` low|high|
    # max is the whole dial; `output_config.format` json_schema documented;
    # caching automatic (no cache_control block documented); tool_choice
    # auto|any|none.  Live 2026-09-03 (receipts/2026-09-03-moonshotai-
    # anthropic): `thinking: {type: disabled}` is honoured; `medium` and
    # `bogus` efforts are accepted silently (allowlist); thinking blocks come
    # back with `signature: ""` and an unsigned block is accepted back
    # (replay "unsigned"); temperature and top_k are swallowed silently
    # (reject); `disable_parallel_tool_use` is not in the schema (reject,
    # the DeepSeek precedent).
    "moonshotai": AnthropicCompat(
        thinking_format="effort",
        thinking_replay="unsigned",
        cache_control="none",
        structured_output="send",
        parallel_tool_calls="reject",
        sampling_params="reject",
        reasoning_efforts=("low", "high", "max"),
        model_prefixes=("kimi-",),
        tool_result_media="images",  # MAP-10: images received (kimi-k3); document block 400 "unsupported content type"
    ),
}

# Default base URLs for the Anthropic presets that name a server.
ANTHROPIC_PRESET_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    # api-docs.deepseek.com first-call.md: base_url https://api.deepseek.com/anthropic,
    # which the Anthropic SDK completes to /anthropic/v1/messages (both the
    # /anthropic and /anthropic/v1 roots answer, live 2026-09-03).
    "deepseek": "https://api.deepseek.com/anthropic/v1",
    # dev.meta.ai protocols--messages.md: base host https://api.meta.ai, the
    # SDK appends /v1/messages — so the /v1 root, like the other two wires.
    "meta": "https://api.meta.ai/v1",
    # platform.kimi.ai api--overview.md: base_url https://api.moonshot.ai/anthropic,
    # endpoint /anthropic/v1/messages — so the /anthropic/v1 root.
    "moonshotai": "https://api.moonshot.ai/anthropic/v1",
}


@dataclass(frozen=True, slots=True)
class ResolvedAnthropicCompat:
    """Fully resolved Anthropic Messages compatibility policy."""

    thinking_format: Literal["anthropic", "deepseek", "adaptive", "effort"] = "anthropic"
    thinking_replay: Literal["signed", "unsigned"] = field(default="signed", kw_only=True)
    cache_control: Literal["anthropic", "none"] = "anthropic"
    structured_output: Literal["send", "reject"] = "send"
    parallel_tool_calls: Literal["send", "reject"] = "send"
    sampling_params: Literal["send", "reject"] = field(default="send", kw_only=True)
    # tool_result.content takes image and document blocks (SDK ToolResultBlockParam;
    # anthropic/claude-code exact 2026-09-07): native unless a server proved otherwise.
    tool_result_media: Literal["native", "images", "reject"] = field(default="native", kw_only=True)
    reasoning_efforts: tuple[str, ...] | None = field(default=None, kw_only=True)
    model_prefixes: tuple[str, ...] | None = None
    extensions: JsonObject | None = None


_ANTHROPIC_AUTO_DEFAULTS: dict[str, str] = {
    "thinking_format": "anthropic",
    "thinking_replay": "signed",
    "cache_control": "anthropic",
    "structured_output": "send",
    "parallel_tool_calls": "send",
    "sampling_params": "send",
    "tool_result_media": "native",
}


def resolve_anthropic_compat(partial: AnthropicCompat) -> ResolvedAnthropicCompat:
    """Resolve a partial Anthropic compat object into concrete serializer policy."""
    kwargs: dict[str, object] = {}
    for field_name, default in _ANTHROPIC_AUTO_DEFAULTS.items():
        value = getattr(partial, field_name)
        kwargs[field_name] = default if value in {None, "auto"} else value
    return ResolvedAnthropicCompat(
        reasoning_efforts=partial.reasoning_efforts,
        model_prefixes=partial.model_prefixes,
        extensions=partial.extensions,
        **kwargs,  # type: ignore[arg-type]
    )


CompatProfile: TypeAlias = OpenAIResponsesCompat | OpenAIChatCompat | AnthropicCompat


# ─── Merge helpers ──────────────────────────────────────────────────


def merge_openai_responses_compat(
    base: OpenAIResponsesCompat,
    override: OpenAIResponsesCompat | None,
) -> OpenAIResponsesCompat:
    """Merge partial OpenAI Responses compat objects.

    None fields inherit. Non-None fields, including "auto", override.
    """
    if override is None:
        return base

    kwargs = {}
    for f in fields(OpenAIResponsesCompat):
        value = getattr(override, f.name)
        if f.name == "extensions":
            kwargs[f.name] = _merge_json_object(base.extensions, override.extensions)
        else:
            kwargs[f.name] = getattr(base, f.name) if value is None else value
    return OpenAIResponsesCompat(**kwargs)


def resolve_openai_responses_compat(partial: OpenAIResponsesCompat) -> ResolvedOpenAIResponsesCompat:
    """Resolve a partial compat object into concrete serializer policy."""
    developer_role = partial.developer_role
    if developer_role in {None, "auto"}:
        developer_role = "developer"

    max_field = partial.max_output_tokens_field
    if max_field in {None, "auto"}:
        max_field = "max_output_tokens"

    reasoning_format = partial.reasoning_format
    if reasoning_format in {None, "auto"}:
        reasoning_format = "responses_reasoning"

    tool_result_name = partial.tool_result_name
    if tool_result_name in {None, "auto"}:
        tool_result_name = "omit"

    strict_tools = partial.strict_tools
    if strict_tools in {None, "auto"}:
        strict_tools = "omit"

    cache_control = partial.cache_control
    if cache_control in {None, "auto"}:
        cache_control = "openai"

    commentary_phase = partial.commentary_phase
    if commentary_phase in {None, "auto"}:
        commentary_phase = "omit"

    edit_image_field = partial.edit_image_field
    if edit_image_field in {None, "auto"}:
        edit_image_field = "array"

    builtin_tools = partial.builtin_tools
    if builtin_tools in {None, "auto"}:
        builtin_tools = "openai"

    tool_result_media = partial.tool_result_media
    if tool_result_media in {None, "auto"}:
        tool_result_media = "native"

    return ResolvedOpenAIResponsesCompat(
        developer_role=developer_role,  # type: ignore[arg-type]
        max_output_tokens_field=max_field,  # type: ignore[arg-type]
        reasoning_format=reasoning_format,  # type: ignore[arg-type]
        tool_result_name=tool_result_name,  # type: ignore[arg-type]
        strict_tools=strict_tools,  # type: ignore[arg-type]
        cache_control=cache_control,  # type: ignore[arg-type]
        commentary_phase=commentary_phase,  # type: ignore[arg-type]
        edit_image_field=edit_image_field,  # type: ignore[arg-type]
        builtin_tools=builtin_tools,  # type: ignore[arg-type]
        tool_result_media=tool_result_media,  # type: ignore[arg-type]
        routing=partial.routing,
        extensions=partial.extensions,
    )
