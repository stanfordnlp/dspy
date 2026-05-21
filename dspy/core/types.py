"""Normalized request, response, and stream types for DSPy language models."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field as dataclass_field
from pprint import pformat
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMBasePart(BaseModel):
    """A single content item in an LM message or output."""

    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class LMTextPart(LMBasePart):
    """Text content."""

    type: Literal["text"] = "text"
    text: str


class LMImagePart(LMBasePart):
    """Image content from data, a URL, a file ID, or a local path."""

    type: Literal["image"] = "image"
    media_type: str = "image/png"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None
    detail: Literal["low", "high", "auto"] | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> "LMImagePart":
        _validate_one_source(self, "LMImagePart")
        return self


class LMAudioPart(LMBasePart):
    """Audio content from data, a URL, a file ID, or a local path."""

    type: Literal["audio"] = "audio"
    media_type: str = "audio/wav"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> "LMAudioPart":
        _validate_one_source(self, "LMAudioPart")
        return self


class LMVideoPart(LMBasePart):
    """Video content from data, a URL, a file ID, or a local path."""

    type: Literal["video"] = "video"
    media_type: str = "video/mp4"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> "LMVideoPart":
        _validate_one_source(self, "LMVideoPart")
        return self


class LMDocumentPart(LMBasePart):
    """Semantic source/document content, optionally citation-enabled.

    Documents are source material: text, PDFs, reports, contracts, or other
    provider-addressable evidence that benefits from title/context/citation
    semantics. Use `LMBinaryPart` for opaque attachments or arbitrary bytes.
    """

    type: Literal["document"] = "document"
    media_type: str = "application/pdf"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None
    source: dict[str, Any] | None = None
    citations: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    context: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "LMDocumentPart":
        has_media_source = any(value is not None for value in (self.data, self.url, self.file_id, self.path))
        if self.source is not None and has_media_source:
            raise ValueError("LMDocumentPart accepts either source or one of data, url, file_id, or path, not both.")
        if self.source is None:
            _validate_one_source(self, "LMDocumentPart")
        elif not self.source:
            raise ValueError("LMDocumentPart.source must be non-empty when provided.")
        return self


class LMBinaryPart(LMBasePart):
    """Opaque binary content from data, a URL, a file ID, or a local path."""

    type: Literal["binary"] = "binary"
    media_type: str = "application/octet-stream"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: str | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> "LMBinaryPart":
        _validate_one_source(self, "LMBinaryPart")
        return self


class LMToolCallPart(LMBasePart):
    """A model request to call a tool.

    Use `dspy.ToolCall(...)` as a shorter public alias when constructing
    assistant messages by hand.

    Args:
        id: Provider call ID, when the backend uses one.
        name: Name of the tool to call.
        args: JSON-like arguments for the tool.
        provider_data: Raw provider fields to keep with the tool call.

    Examples:
        ```python
        import dspy

        assistant = dspy.Assistant(
            dspy.ToolCall(
                id="call_1",
                name="search",
                args={"query": "DSPy"},
            )
        )
        ```
    """

    type: Literal["tool_call"] = "tool_call"
    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)


class LMToolResultPart(LMBasePart):
    """A tool execution result sent back to a model."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str | None = None
    name: str | None = None
    content: list["LMPart"] = Field(default_factory=list)
    is_error: bool = False
    provider_data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_content(cls, data: Any) -> Any:
        if isinstance(data, dict) and "content" in data:
            data = dict(data)
            content = data["content"]
            if isinstance(content, list):
                data["content"] = [_coerce_part(item) for item in content]
            else:
                data["content"] = [_coerce_part(content)]
        return data


class LMThinkingPart(LMBasePart):
    """Reasoning or thinking content returned by a model."""

    type: Literal["thinking"] = "thinking"
    text: str
    redacted: bool = False


class LMCitationPart(LMBasePart):
    """A source citation returned by a model."""

    type: Literal["citation"] = "citation"
    text: str | None = None
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_has_content(self) -> "LMCitationPart":
        if self.text is None and self.title is None and self.url is None:
            raise ValueError("LMCitationPart requires at least one of text, title, or url.")
        return self


class LMRefusalPart(LMBasePart):
    """A model refusal."""

    type: Literal["refusal"] = "refusal"
    text: str


LMPart = Annotated[
    LMTextPart
    | LMImagePart
    | LMAudioPart
    | LMVideoPart
    | LMDocumentPart
    | LMBinaryPart
    | LMToolCallPart
    | LMToolResultPart
    | LMThinkingPart
    | LMCitationPart
    | LMRefusalPart,
    Field(discriminator="type"),
]


class LMMessage(BaseModel):
    """A role-attributed sequence of LM parts."""

    role: str
    parts: list[LMPart]
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_parts(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            data = dict(data)
            if "parts" not in data and "content" in data:
                data["parts"] = _parts_from_openai_content(data.pop("content"))
            elif "parts" in data:
                data["parts"] = [_coerce_part(part) for part in data["parts"]]
        return data

    @property
    def text(self) -> str | None:
        texts = [part.text for part in self.parts if isinstance(part, LMTextPart)]
        return "".join(texts) if texts else None


class LMToolSpec(BaseModel):
    """Provider-independent schema for a tool available to an LM."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class LMReasoningConfig(BaseModel):
    """Reasoning controls for models with native reasoning support."""

    effort: str | None = None
    max_tokens: int | None = None
    summary: str | None = None


class LMToolChoice(BaseModel):
    """Tool-choice controls for native tool-capable models."""

    mode: Literal["auto", "required", "none"] = "auto"
    allowed: list[str] = Field(default_factory=list)
    parallel: bool | None = None


class LMCacheConfig(BaseModel):
    """DSPy memoization cache controls for a normalized LM request.

    This cache skips the provider call entirely when DSPy finds an exact
    request match. Use `LMPromptCacheConfig` for provider-side prompt/token
    caching that still sends the request to the provider.
    """

    enabled: bool | None = None
    rollout_id: int | str | None = None


class LMPromptCacheConfig(BaseModel):
    """Provider-side prompt/token cache controls.

    Prompt caching is not DSPy memoization. The provider call still happens,
    but the backend may reuse cached prompt prefixes or KV state for lower
    latency or lower input-token cost.
    """

    enabled: bool | None = None
    key: str | None = None


_KNOWN_CONFIG_KEYS = {
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "n",
    "logprobs",
    "response_format",
    "reasoning",
    "reasoning_effort",
    "tool_choice",
    "parallel_tool_calls",
    "cache",
    "rollout_id",
    "prompt_cache",
    "prompt_cache_key",
    "extensions",
}


class LMConfig(BaseModel):
    """Common generation controls for an LM request."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = Field(default_factory=list)
    n: int | None = None
    logprobs: bool | int | None = None
    response_format: Any | None = None
    reasoning: LMReasoningConfig | None = None
    tool_choice: LMToolChoice | None = None
    cache: LMCacheConfig | None = None
    prompt_cache: LMPromptCacheConfig | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "LMConfig":
        data: dict[str, Any] = {}
        extensions = dict(kwargs.pop("extensions", {}) or {})

        for key, value in kwargs.items():
            if key in _KNOWN_CONFIG_KEYS:
                data[key] = value
            else:
                extensions[key] = value

        if "reasoning_effort" in data:
            effort = data.pop("reasoning_effort")
            data["reasoning"] = data.get("reasoning") or LMReasoningConfig(effort=effort)
        if isinstance(data.get("reasoning"), dict):
            data["reasoning"] = LMReasoningConfig(**data["reasoning"])

        parallel = data.pop("parallel_tool_calls", None)
        if "tool_choice" in data:
            choice = data["tool_choice"]
            if isinstance(choice, str):
                data["tool_choice"] = LMToolChoice(mode=choice, parallel=parallel)
            elif isinstance(choice, dict):
                data["tool_choice"] = LMToolChoice(**({"parallel": parallel} | choice))
            elif isinstance(choice, LMToolChoice) and parallel is not None:
                data["tool_choice"] = choice.model_copy(update={"parallel": parallel})
        elif parallel is not None:
            data["tool_choice"] = LMToolChoice(parallel=parallel)

        cache = data.pop("cache", None) if "cache" in data else None
        rollout_id = data.pop("rollout_id", None) if "rollout_id" in data else None
        if cache is not None or rollout_id is not None:
            if isinstance(cache, LMCacheConfig):
                data["cache"] = cache.model_copy(update={"rollout_id": rollout_id}) if rollout_id is not None else cache
            else:
                data["cache"] = LMCacheConfig(enabled=cache, rollout_id=rollout_id)

        prompt_cache = data.pop("prompt_cache", None) if "prompt_cache" in data else None
        prompt_cache_key = data.pop("prompt_cache_key", None) if "prompt_cache_key" in data else None
        if prompt_cache is not None or prompt_cache_key is not None:
            if isinstance(prompt_cache, LMPromptCacheConfig):
                if prompt_cache_key is not None:
                    prompt_cache = prompt_cache.model_copy(update={"key": prompt_cache_key})
                data["prompt_cache"] = prompt_cache
            else:
                data["prompt_cache"] = LMPromptCacheConfig(enabled=prompt_cache, key=prompt_cache_key)

        data["extensions"] = extensions
        return cls(**data)


def _merge_lm_config(left: LMConfig | None, right: LMConfig | None) -> LMConfig | None:
    if left is None:
        return right
    if right is None:
        return left

    data = left.model_dump()
    right_data = right.model_dump(exclude_none=True)
    extensions = {**left.extensions, **right.extensions}
    data.update(right_data)
    data["extensions"] = extensions
    return LMConfig(**data)


def _merge_config_overrides(config: LMConfig, kwargs: dict[str, Any]) -> LMConfig:
    data = config.model_dump()
    extensions = dict(config.extensions)

    direct_keys = {
        "temperature",
        "max_tokens",
        "top_p",
        "stop",
        "n",
        "logprobs",
        "response_format",
    }
    for key in direct_keys & kwargs.keys():
        data[key] = kwargs[key]

    if "reasoning" in kwargs:
        reasoning = kwargs["reasoning"]
        if isinstance(reasoning, dict):
            reasoning = LMReasoningConfig(**reasoning)
        data["reasoning"] = reasoning
    if "reasoning_effort" in kwargs:
        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning = LMReasoningConfig(**reasoning)
        reasoning = reasoning or LMReasoningConfig()
        data["reasoning"] = reasoning.model_copy(update={"effort": kwargs["reasoning_effort"]})

    if "tool_choice" in kwargs:
        choice = kwargs["tool_choice"]
        if isinstance(choice, str):
            choice = LMToolChoice(mode=choice)
        elif isinstance(choice, dict):
            choice = LMToolChoice(**choice)
        data["tool_choice"] = choice
    if "parallel_tool_calls" in kwargs:
        choice = data.get("tool_choice")
        if isinstance(choice, dict):
            choice = LMToolChoice(**choice)
        choice = choice or LMToolChoice()
        data["tool_choice"] = choice.model_copy(update={"parallel": kwargs["parallel_tool_calls"]})

    if "cache" in kwargs or "rollout_id" in kwargs:
        cache = data.get("cache")
        if isinstance(cache, dict):
            cache = LMCacheConfig(**cache)
        if isinstance(kwargs.get("cache"), LMCacheConfig):
            cache = kwargs["cache"]
        else:
            cache = cache or LMCacheConfig()
            update = {}
            if "cache" in kwargs:
                update["enabled"] = kwargs["cache"]
            if "rollout_id" in kwargs:
                update["rollout_id"] = kwargs["rollout_id"]
            cache = cache.model_copy(update=update)
        data["cache"] = cache

    if "prompt_cache" in kwargs or "prompt_cache_key" in kwargs:
        prompt_cache = data.get("prompt_cache")
        if isinstance(prompt_cache, dict):
            prompt_cache = LMPromptCacheConfig(**prompt_cache)
        if isinstance(kwargs.get("prompt_cache"), LMPromptCacheConfig):
            prompt_cache = kwargs["prompt_cache"]
        else:
            prompt_cache = prompt_cache or LMPromptCacheConfig()
            update = {}
            if "prompt_cache" in kwargs:
                update["enabled"] = kwargs["prompt_cache"]
            if "prompt_cache_key" in kwargs:
                update["key"] = kwargs["prompt_cache_key"]
            prompt_cache = prompt_cache.model_copy(update=update)
        data["prompt_cache"] = prompt_cache

    if "extensions" in kwargs:
        extra = kwargs["extensions"]
        if extra is None:
            extensions = {}
        elif isinstance(extra, Mapping):
            extensions.update(extra)
        else:
            raise TypeError("`extensions` override must be a mapping or None.")

    handled = _KNOWN_CONFIG_KEYS
    for key, value in kwargs.items():
        if key not in handled:
            extensions[key] = value
    data["extensions"] = extensions
    return LMConfig(**data)


@dataclass
class LMRequestPatch:
    """A partial normalized LM request contributed while rendering a DSPy call.

    `LMRequest` is the complete object a `LanguageModel` receives. A patch is
    the smaller, composable unit that DSPy type strategies can contribute while
    an adapter is still building that request: extra messages, extra parts,
    native tools, native config, or signature fields that should be hidden from
    the outer adapter's ordinary text/JSON/XML rendering.
    """

    messages: list[LMMessage] = dataclass_field(default_factory=list)
    system_parts: list[LMPart] = dataclass_field(default_factory=list)
    user_parts: list[LMPart] = dataclass_field(default_factory=list)
    assistant_parts: list[LMPart] = dataclass_field(default_factory=list)
    tools: list[LMToolSpec] = dataclass_field(default_factory=list)
    config: LMConfig | None = None
    delete_input_fields: tuple[str, ...] = ()
    delete_output_fields: tuple[str, ...] = ()
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def merge(self, other: "LMRequestPatch") -> "LMRequestPatch":
        """Return a new patch containing this patch followed by `other`."""
        return LMRequestPatch(
            messages=[*self.messages, *other.messages],
            system_parts=[*self.system_parts, *other.system_parts],
            user_parts=[*self.user_parts, *other.user_parts],
            assistant_parts=[*self.assistant_parts, *other.assistant_parts],
            tools=[*self.tools, *other.tools],
            config=_merge_lm_config(self.config, other.config),
            delete_input_fields=(*self.delete_input_fields, *other.delete_input_fields),
            delete_output_fields=(*self.delete_output_fields, *other.delete_output_fields),
            metadata={**self.metadata, **other.metadata},
        )

    def as_lm_kwargs(self) -> dict[str, Any]:
        """Return the legacy kwargs implied by this patch.

        This keeps the first implementation usable with today's adapter call
        path, which still passes `lm_kwargs` rather than an `LMRequestPatch` all
        the way down. Message and part patches are intentionally not flattened
        here; they require the next adapter-call refactor.
        """
        kwargs = self.config.model_dump(exclude_none=True) if self.config is not None else {}
        if self.tools:
            kwargs["tools"] = list(self.tools)
        return kwargs


class LMRequest(BaseModel):
    """A normalized request passed to a `LanguageModel`."""

    model: str
    messages: list[LMMessage]
    tools: list[LMToolSpec] = Field(default_factory=list)
    config: LMConfig = Field(default_factory=LMConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def from_call(
        cls,
        *,
        model: str,
        items: tuple[Any, ...] = (),
        prompt: str | None = None,
        messages: list[dict[str, Any] | LMMessage] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> "LMRequest":
        if messages is not None and (items or prompt is not None):
            raise ValueError("Pass messages or direct-call inputs, not both.")

        collected_tools: list[Any] = list(tools or [])
        if messages is not None:
            normalized_messages = [_coerce_message(message) for message in messages]
        else:
            normalized_messages, positional_tools = _messages_from_items(items, prompt=prompt)
            collected_tools.extend(positional_tools)

        config = LMConfig.from_kwargs(**kwargs)
        return cls(
            model=model,
            messages=normalized_messages,
            tools=[_coerce_tool_spec(tool) for tool in collected_tools],
            config=config,
        )

    @classmethod
    def from_prompt_or_messages(
        cls,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict[str, Any] | LMMessage] | None = None,
        **kwargs: Any,
    ) -> "LMRequest":
        return cls.from_call(model=model, prompt=prompt, messages=messages, **kwargs)

    def with_config_overrides(self, **kwargs: Any) -> "LMRequest":
        """Return a copy with explicit request config overrides applied.

        Only fields implied by the supplied keyword arguments are changed. This
        preserves existing grouped config such as `cache`, `prompt_cache`,
        `tool_choice`, `reasoning`, `stop`, and provider-specific
        `extensions` when an unrelated setting is overridden.
        """
        if not kwargs:
            return self
        merged = _merge_config_overrides(self.config, kwargs)
        return self.model_copy(update={"config": merged}, deep=True)


class LMUsage(BaseModel):
    """Token and timing usage for one LM request."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def fill_aliases(self) -> "LMUsage":
        if self.input_tokens is None and self.prompt_tokens is not None:
            self.input_tokens = self.prompt_tokens
        if self.output_tokens is None and self.completion_tokens is not None:
            self.output_tokens = self.completion_tokens
        if self.prompt_tokens is None and self.input_tokens is not None:
            self.prompt_tokens = self.input_tokens
        if self.completion_tokens is None and self.output_tokens is not None:
            self.completion_tokens = self.output_tokens
        if self.total_tokens is None and self.input_tokens is not None and self.output_tokens is not None:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


class LMOutput(BaseModel):
    """One generated candidate in an LM response."""

    parts: list[LMPart] = Field(default_factory=list)
    finish_reason: str | None = None
    truncated: bool = False
    logprobs: Any | None = None
    provider_output: Any | None = Field(default=None, exclude=True)
    provider_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_parts(cls, data: Any) -> Any:
        if isinstance(data, dict) and "parts" in data:
            data = dict(data)
            data["parts"] = [_coerce_part(part) for part in data["parts"]]
        return data

    @property
    def text(self) -> str | None:
        texts = [part.text for part in self.parts if isinstance(part, LMTextPart)]
        return "".join(texts) if texts else None

    @property
    def reasoning_content(self) -> str | None:
        texts = [part.text for part in self.parts if isinstance(part, LMThinkingPart)]
        return "".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[LMToolCallPart]:
        return [part for part in self.parts if isinstance(part, LMToolCallPart)]

    @property
    def citations(self) -> list[LMCitationPart]:
        return [part for part in self.parts if isinstance(part, LMCitationPart)]

    @property
    def images(self) -> list[LMImagePart]:
        return [part for part in self.parts if isinstance(part, LMImagePart)]

    @property
    def audio(self) -> list[LMAudioPart]:
        return [part for part in self.parts if isinstance(part, LMAudioPart)]

    @property
    def videos(self) -> list[LMVideoPart]:
        return [part for part in self.parts if isinstance(part, LMVideoPart)]

    @property
    def documents(self) -> list[LMDocumentPart]:
        return [part for part in self.parts if isinstance(part, LMDocumentPart)]

    @property
    def binaries(self) -> list[LMBinaryPart]:
        return [part for part in self.parts if isinstance(part, LMBinaryPart)]

    @property
    def refusal(self) -> str | None:
        refusals = [part.text for part in self.parts if isinstance(part, LMRefusalPart)]
        return "".join(refusals) if refusals else None

    def to_value(self) -> Any:
        values = [_part_to_value(part) for part in self.parts]
        values = [value for value in values if value is not None]
        if len(values) == 1 and isinstance(values[0], str) and self.logprobs is None:
            return values[0]
        return values

    def to_output_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text}
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            data["tool_calls"] = [_tool_call_to_provider_dict(call) for call in self.tool_calls]
        if self.citations:
            data["citations"] = [citation.model_dump(exclude_none=True) for citation in self.citations]
        if self.logprobs is not None:
            data["logprobs"] = self.logprobs
        return data


class LMResponse(BaseModel):
    """The normalized result of one LM request."""

    model: str | None = None
    outputs: list[LMOutput]
    usage: LMUsage | dict[str, Any] | None = None
    cost: float | None = None
    cache_hit: bool = False
    response_id: str | None = None
    provider_response: Any | None = Field(default=None, exclude=True)
    provider_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_usage(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("usage"), dict):
            data = dict(data)
            data["usage"] = LMUsage(**data["usage"])
        return data

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        model: str | None = None,
        usage: LMUsage | dict[str, Any] | None = None,
        cost: float | None = None,
        cache_hit: bool = False,
        **kwargs: Any,
    ) -> "LMResponse":
        return cls(
            model=model,
            outputs=[LMOutput(parts=[LMTextPart(text=text)])],
            usage=usage,
            cost=cost,
            cache_hit=cache_hit,
            **kwargs,
        )

    def __iter__(self):
        return iter(self.to_values())

    def __getitem__(self, index: int) -> Any:
        return self.to_values()[index]

    def __len__(self) -> int:
        return len(self.outputs)

    @property
    def output(self) -> LMOutput:
        return self.outputs[0]

    @property
    def parts(self) -> list[LMPart]:
        return self.output.parts

    @property
    def text(self) -> str | None:
        return self.output.text

    @property
    def reasoning_content(self) -> str | None:
        return self.output.reasoning_content

    @property
    def tool_calls(self) -> list[LMToolCallPart]:
        return self.output.tool_calls

    @property
    def citations(self) -> list[LMCitationPart]:
        return self.output.citations

    @property
    def images(self) -> list[LMImagePart]:
        return self.output.images

    @property
    def audio(self) -> list[LMAudioPart]:
        return self.output.audio

    @property
    def videos(self) -> list[LMVideoPart]:
        return self.output.videos

    @property
    def documents(self) -> list[LMDocumentPart]:
        return self.output.documents

    @property
    def binaries(self) -> list[LMBinaryPart]:
        return self.output.binaries

    def to_values(self) -> list[Any]:
        return [output.to_value() for output in self.outputs]

    def to_outputs(self) -> list[Any]:
        outputs: list[Any] = []
        for output in self.outputs:
            if _requires_output_dict(output):
                outputs.append(output.to_output_dict())
            else:
                outputs.append(output.to_value())
        return outputs

    def to_legacy_outputs(self) -> list[Any]:
        return self.to_outputs()

    def usage_as_dict(self) -> dict[str, Any]:
        if self.usage is None:
            return {}
        if isinstance(self.usage, LMUsage):
            return self.usage.model_dump(exclude_none=True)
        return dict(self.usage)


class LMHistoryEntry(BaseModel, Mapping[str, Any]):
    """A typed history record that can be read like a dictionary.

    Store the canonical request and response, then derive legacy convenience
    fields such as `outputs`, `usage`, `messages`, and `kwargs` on demand.
    Because this class implements `Mapping`, existing history code can keep
    using `entry["messages"]`, `entry.get("prompt")`, `entry.items()`, and
    `dict(entry)`.
    """

    request: LMRequest
    response: LMResponse
    timestamp: str
    uuid: str
    model_type: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def drop_derived_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in _HISTORY_DERIVED_KEYS:
                data.pop(key, None)
        return data

    @property
    def outputs(self) -> list[Any]:
        return self.response.to_outputs()

    @property
    def usage(self) -> dict[str, Any]:
        return self.response.usage_as_dict()

    @property
    def cost(self) -> float | None:
        return self.response.cost

    @property
    def model(self) -> str:
        return self.request.model

    @property
    def prompt(self) -> str | None:
        return _history_request_prompt(self.request)

    @property
    def messages(self) -> list[dict[str, Any]] | None:
        return _history_request_messages_as_openai(self.request)

    @property
    def kwargs(self) -> dict[str, Any]:
        return _history_request_kwargs(self.request)

    @property
    def response_model(self) -> str | None:
        return self.response.model

    def __getitem__(self, key: str) -> Any:
        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())

    def __repr__(self) -> str:
        formatted = pformat(self._essential_mapping(), width=100, sort_dicts=False)
        return f"LMHistoryEntry(\n{formatted}\n)"

    def __str__(self) -> str:
        return repr(self)

    def to_dict(self, *, mode: str = "python", exclude_none: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Return this history entry as a plain dictionary."""
        if kwargs:
            return self.model_dump(mode=mode, exclude_none=exclude_none, **kwargs)
        data = self._mapping()
        if mode != "python":
            data = json.loads(json.dumps(data, default=_json_default))
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data

    def _essential_mapping(self) -> dict[str, Any]:
        data = self.model_dump(mode="python", exclude_none=True)
        data.update(self.model_extra or {})
        return data

    def _mapping(self) -> dict[str, Any]:
        data = self._essential_mapping()
        data.update({key: getattr(self, key) for key in _HISTORY_DERIVED_KEYS})
        return {key: value for key, value in data.items() if value is not None}


_HISTORY_DERIVED_KEYS = (
    "outputs",
    "usage",
    "cost",
    "model",
    "prompt",
    "messages",
    "kwargs",
    "response_model",
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return str(value)


class LMDelta(BaseModel):
    """Base class for streamed content deltas."""

    type: str


class LMTextDelta(LMDelta):
    type: Literal["text_delta"] = "text_delta"
    text: str


class LMThinkingDelta(LMDelta):
    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class LMToolCallDelta(LMDelta):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str | None = None
    name: str | None = None
    args_delta: str | None = None


class LMCitationDelta(LMDelta):
    type: Literal["citation_delta"] = "citation_delta"
    citation: LMCitationPart


class LMImageDelta(LMDelta):
    type: Literal["image_delta"] = "image_delta"
    image: LMImagePart


class LMAudioDelta(LMDelta):
    type: Literal["audio_delta"] = "audio_delta"
    audio: LMAudioPart


LMAnyDelta = Annotated[
    LMTextDelta | LMThinkingDelta | LMToolCallDelta | LMCitationDelta | LMImageDelta | LMAudioDelta,
    Field(discriminator="type"),
]


class LMStreamEvent(BaseModel):
    """Base class for normalized LM stream events."""

    type: str


class LMStreamStartEvent(LMStreamEvent):
    type: Literal["start"] = "start"
    model: str | None = None


class LMStreamDeltaEvent(LMStreamEvent):
    type: Literal["delta"] = "delta"
    output_index: int = 0
    part_index: int
    delta: LMAnyDelta


class LMStreamOutputEndEvent(LMStreamEvent):
    type: Literal["output_end"] = "output_end"
    output_index: int = 0
    finish_reason: str | None = None
    truncated: bool = False


class LMStreamEndEvent(LMStreamEvent):
    type: Literal["end"] = "end"
    usage: LMUsage | dict[str, Any] | None = None
    cost: float | None = None
    response: LMResponse | None = None


class LMStreamErrorEvent(LMStreamEvent):
    type: Literal["error"] = "error"
    error: Exception

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LMOutputBuilder:
    """Assemble streamed LM events into a final `LMResponse`."""

    def __init__(self):
        self.model: str | None = None
        self._parts: dict[int, list[LMPart | None]] = {}
        self._finish_reasons: dict[int, str | None] = {}
        self._truncated: dict[int, bool] = {}

    def apply(self, event: LMStreamEvent) -> LMResponse | None:
        if isinstance(event, LMStreamStartEvent):
            self.model = event.model
            return None
        if isinstance(event, LMStreamDeltaEvent):
            self._apply_delta(event)
            return None
        if isinstance(event, LMStreamOutputEndEvent):
            self._finish_reasons[event.output_index] = event.finish_reason
            self._truncated[event.output_index] = event.truncated
            return None
        if isinstance(event, LMStreamEndEvent):
            if event.response is not None:
                return event.response
            return self.to_response(usage=event.usage, cost=event.cost)
        if isinstance(event, LMStreamErrorEvent):
            raise event.error
        return None

    def to_response(self, *, usage: LMUsage | dict[str, Any] | None = None, cost: float | None = None) -> LMResponse:
        max_index = max(self._parts.keys(), default=0)
        outputs = []
        for output_index in range(max_index + 1):
            parts = [part for part in self._parts.get(output_index, []) if part is not None]
            outputs.append(
                LMOutput(
                    parts=parts,
                    finish_reason=self._finish_reasons.get(output_index),
                    truncated=self._truncated.get(output_index, False),
                )
            )
        return LMResponse(model=self.model, outputs=outputs, usage=usage, cost=cost)

    def _apply_delta(self, event: LMStreamDeltaEvent) -> None:
        parts = self._parts.setdefault(event.output_index, [])
        while len(parts) <= event.part_index:
            parts.append(None)

        current = parts[event.part_index]
        delta = event.delta
        if isinstance(delta, LMThinkingDelta):
            text = (current.text if isinstance(current, LMThinkingPart) else "") + delta.text
            parts[event.part_index] = LMThinkingPart(text=text)
        elif isinstance(delta, LMTextDelta):
            text = (current.text if isinstance(current, LMTextPart) else "") + delta.text
            parts[event.part_index] = LMTextPart(text=text)
        elif isinstance(delta, LMToolCallDelta):
            buffer = ""
            if isinstance(current, LMToolCallPart):
                buffer = current.provider_data.get("args_buffer", "")
            buffer += delta.args_delta or ""
            args = _parse_json_object(buffer)
            parts[event.part_index] = LMToolCallPart(
                id=delta.id if delta.id is not None else getattr(current, "id", None),
                name=delta.name if delta.name is not None else getattr(current, "name", ""),
                args=args,
                provider_data={"args_buffer": buffer},
            )
        elif isinstance(delta, LMCitationDelta):
            parts[event.part_index] = delta.citation
        elif isinstance(delta, LMImageDelta):
            parts[event.part_index] = delta.image
        elif isinstance(delta, LMAudioDelta):
            parts[event.part_index] = delta.audio


class LMStream:
    """Synchronous LM stream with a final `LMResponse` result."""

    def __init__(
        self,
        *,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        finalize: Callable[[LMRequest, LMResponse], LMResponse],
    ):
        self.request = request
        self._events = events
        self._finalize = finalize
        self._builder = LMOutputBuilder()
        self._result: LMResponse | None = None

    def __iter__(self) -> Iterator[LMStreamEvent]:
        for event in self._events:
            response = self._builder.apply(event)
            if response is not None:
                self._result = self._finalize(self.request, response)
            yield event

    def result(self) -> LMResponse:
        if self._result is None:
            raise RuntimeError("Stream has not completed yet.")
        return self._result


class AsyncLMStream:
    """Asynchronous LM stream with a final `LMResponse` result."""

    def __init__(
        self,
        *,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        finalize: Callable[[LMRequest, LMResponse], LMResponse],
    ):
        self.request = request
        self._events = events
        self._finalize = finalize
        self._builder = LMOutputBuilder()
        self._result: LMResponse | None = None

    async def __aiter__(self) -> AsyncIterator[LMStreamEvent]:
        async for event in self._events:
            response = self._builder.apply(event)
            if response is not None:
                self._result = self._finalize(self.request, response)
            yield event

    def result(self) -> LMResponse:
        if self._result is None:
            raise RuntimeError("Stream has not completed yet.")
        return self._result


def System(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    """Create a system message for a direct LM call.

    A system message gives model-level instructions, such as tone, scope, or
    formatting rules. Pass text, media parts, or normalized `LMPart` objects;
    DSPy stores them as one `LMMessage` with role `"system"`.

    Args:
        *parts: Text, DSPy media objects, or normalized LM parts to include in
            the message.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` that can be passed to `dspy.LanguageModel`, `dspy.LM`, or
        `dspy.LMRequest`.

    Examples:
        System instruction with a user turn:

        ```python
        import dspy

        lm = dspy.LanguageModel(model="test/model")
        request = lm.normalize_request(
            dspy.System("You are concise."),
            dspy.User("What is DSPy?"),
        )
        ```

    See Also:
        [`dspy.User`][dspy.User]
        [`dspy.Assistant`][dspy.Assistant]
        [`dspy.LMRequest`][dspy.LMRequest]
    """
    return LMMessage(role="system", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


def Developer(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    """Create a developer message for a direct LM call.

    A developer message carries instructions that sit between system guidance
    and user content. Use it when a provider supports a `"developer"` role and
    you want to keep implementation guidance separate from the user's request.

    Args:
        *parts: Text, DSPy media objects, or normalized LM parts to include in
            the message.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` with role `"developer"`.

    Examples:
        Add house-style instructions:

        ```python
        import dspy

        request = dspy.LMRequest.from_call(
            model="openai/gpt-4o-mini",
            items=(
                dspy.System("You are a technical editor."),
                dspy.Developer("Prefer short examples."),
                dspy.User("Explain callbacks."),
            ),
        )
        ```

    See Also:
        [`dspy.System`][dspy.System]
        [`dspy.User`][dspy.User]
    """
    return LMMessage(role="developer", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


def User(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    """Create a user message for a direct LM call.

    A user message contains the request or data you want the model to answer.
    Pass plain text for simple prompts, or mix text with images, audio, documents,
    binary attachments, and normalized LM parts for multimodal calls.

    Args:
        *parts: Text, DSPy media objects, or normalized LM parts to include in
            the message.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` with role `"user"`.

    Examples:
        Multi-turn LM call:

        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        response = lm(
            dspy.User("What is DSPy?"),
            dspy.Assistant("DSPy is a framework for programming LM pipelines."),
            dspy.User("Say that in five words."),
        )
        ```

        Multi-turn call with media:

        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        response = lm(
            dspy.System("Answer in one sentence."),
            dspy.User(
                "Describe this image.",
                dspy.Image("https://example.com/dog.png"),
            ),
        )
        ```

        For a single user turn, pass the parts directly to `lm(...)` instead:

        ```python
        response = lm("Describe this image.", dspy.Image("https://example.com/dog.png"))
        ```

        Explicit `LMRequest` for custom LM authors and advanced users:

        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        request = dspy.LMRequest(
            model="openai/gpt-4o-mini",
            messages=[
                dspy.System("You are concise."),
                dspy.User(
                    "Describe this image.",
                    dspy.Image("https://example.com/dog.png"),
                ),
            ],
            config=dspy.LMConfig(temperature=0.2, max_tokens=200),
        )

        response = lm(request)
        ```

    See Also:
        [`dspy.System`][dspy.System]
        [`dspy.Assistant`][dspy.Assistant]
        [`dspy.ToolResult`][dspy.ToolResult]
    """
    return LMMessage(role="user", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


def Assistant(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:
    """Create an assistant message for a direct LM call.

    An assistant message represents a previous model response. Use it when you
    build a multi-turn request by hand, or when you send tool calls back to the
    model before adding tool results.

    Args:
        *parts: Text, tool calls, citations, media parts, or other normalized LM
            parts from an assistant turn.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` with role `"assistant"`.

    Examples:
        Continue a conversation:

        ```python
        import dspy

        request = dspy.LMRequest.from_call(
            model="openai/gpt-4o-mini",
            items=(
                dspy.User("What is DSPy?"),
                dspy.Assistant("DSPy is a framework for programming LM pipelines."),
                dspy.User("Say that in five words."),
            ),
        )
        ```

    See Also:
        [`dspy.User`][dspy.User]
        [`dspy.ToolCall`][dspy.ToolCall]
        [`dspy.ToolResult`][dspy.ToolResult]
    """
    return LMMessage(role="assistant", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


ToolCall = LMToolCallPart
"""Create a tool-call part for an assistant message.

`ToolCall` is an alias for `LMToolCallPart`. Use it inside `dspy.Assistant(...)`
when you want to include a model-requested tool call in a normalized
conversation.
"""


def ToolResult(
    *parts: Any,
    call_id: str | None = None,
    name: str | None = None,
    content: Any | None = None,
    is_error: bool = False,
) -> LMMessage:
    """Create a tool-result message for a direct LM call.

    A tool-result message sends the output of a tool back to the model. Pass the
    returned text or media as `*parts` or with `content=...`, and include the
    `call_id` from the matching assistant tool call when the provider uses call
    IDs.

    Args:
        *parts: Text, DSPy media objects, or normalized LM parts returned by the
            tool. If you pass one `LMToolResultPart`, DSPy uses it directly.
        call_id: Identifier of the assistant tool call this result answers.
        name: Tool name associated with the result.
        content: Optional tool output passed by keyword. Use this when adapting
            OpenAI-style code that stores the result under `content`.
        is_error: Whether this result represents a failed tool execution.

    Returns:
        An `LMMessage` with role `"tool"` and one `LMToolResultPart`.

    Examples:
        Send a weather result back to the model:

        ```python
        import dspy

        messages = [
            dspy.User("What is the weather in Paris?"),
            dspy.Assistant(
                dspy.ToolCall(
                    id="call_1",
                    name="get_weather",
                    args={"location": "Paris"},
                )
            ),
            dspy.ToolResult(
                '{"temperature": "22", "unit": "celsius"}',
                call_id="call_1",
                name="get_weather",
            ),
            dspy.User("Summarize the result."),
        ]
        ```

    See Also:
        [`dspy.ToolCall`][dspy.ToolCall]
        [`dspy.Assistant`][dspy.Assistant]
        [`dspy.Tool`][dspy.Tool]
    """
    if content is not None:
        if parts:
            raise TypeError("Pass tool output either as positional parts or as `content=...`, not both.")
        parts = tuple(content if isinstance(content, list) else [content])

    if len(parts) == 1 and isinstance(parts[0], LMToolResultPart):
        result = parts[0]
    else:
        result = LMToolResultPart(
            call_id=call_id,
            name=name,
            content=[_coerce_part(part) for part in parts],
            is_error=is_error,
        )
    return LMMessage(role="tool", parts=[result])


def _history_request_prompt(request: LMRequest) -> str | None:
    if len(request.messages) != 1:
        return None
    message = request.messages[0]
    if message.role != "user" or len(message.parts) != 1:
        return None
    part = message.parts[0]
    return part.text if isinstance(part, LMTextPart) else None


def _history_request_messages_as_openai(request: LMRequest) -> list[dict[str, Any]]:
    messages = []
    for message in request.messages:
        if message.role == "assistant":
            tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
            content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
            item: dict[str, Any] = {
                "role": "assistant",
                "content": _history_message_parts_as_openai_content(content_parts) if content_parts else None,
            }
            if tool_calls:
                item["tool_calls"] = [_history_tool_call_as_openai(call) for call in tool_calls]
        elif message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
            result = message.parts[0]
            item = {"role": "tool", "content": _history_tool_result_content(result)}
            if result.call_id is not None:
                item["tool_call_id"] = result.call_id
            if result.name is not None:
                item["name"] = result.name
        else:
            item = {
                "role": message.role,
                "content": _history_message_parts_as_openai_content(message.parts),
            }
        if message.name is not None and "name" not in item:
            item["name"] = message.name
        messages.append(item)
    return messages


def _history_tool_call_as_openai(call: LMToolCallPart) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.args),
        },
    }
    if call.id is not None:
        data["id"] = call.id
    return data


def _history_tool_result_content(result: LMToolResultPart) -> str:
    chunks = []
    for part in result.content:
        if isinstance(part, LMTextPart):
            chunks.append(part.text)
        else:
            chunks.append(json.dumps(part.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
    return "".join(chunks)


def _history_message_parts_as_openai_content(parts: list[LMPart]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and isinstance(parts[0], LMTextPart):
        return parts[0].text
    return [_history_part_as_openai_content(part) for part in parts]


def _history_part_as_openai_content(part: LMPart) -> dict[str, Any]:
    if isinstance(part, LMTextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, LMImagePart):
        return {"type": "image_url", "image_url": {"url": _history_part_source(part)}}
    if isinstance(part, LMAudioPart):
        return {
            "type": "input_audio",
            "input_audio": {"data": part.data, "format": _history_media_format(part.media_type)},
        }
    if isinstance(part, LMVideoPart):
        return {"type": "video", "video": {"url": _history_part_source(part), "media_type": part.media_type}}
    if isinstance(part, LMDocumentPart):
        data = {"type": "document"}
        if part.source is not None:
            data["source"] = part.source
        else:
            data["source"] = _history_part_source(part)
            data["media_type"] = part.media_type
        if part.citations:
            data["citations"] = part.citations
        if part.title is not None:
            data["title"] = part.title
        if part.context is not None:
            data["context"] = part.context
        return data
    if isinstance(part, LMBinaryPart):
        return {
            "type": "binary",
            "binary": {
                key: value
                for key, value in {
                    "data": _history_part_source(part),
                    "file_id": part.file_id,
                    "filename": part.filename,
                    "media_type": part.media_type,
                }.items()
                if value is not None
            },
        }
    return part.model_dump(exclude_none=True)


def _history_part_source(part: LMImagePart | LMAudioPart | LMVideoPart | LMDocumentPart | LMBinaryPart) -> str | None:
    if part.data is not None:
        return part.data if part.data.startswith("data:") else f"data:{part.media_type};base64,{part.data}"
    return part.url or part.file_id or part.path


def _history_media_format(media_type: str) -> str:
    return media_type.split("/", 1)[1] if "/" in media_type else media_type


def _history_request_kwargs(request: LMRequest) -> dict[str, Any]:
    return request.config.model_dump(exclude_none=True)


def _validate_one_source(part: Any, class_name: str) -> None:
    sources = [part.data, part.url, part.file_id, part.path]
    if sum(source is not None for source in sources) != 1:
        raise ValueError(f"{class_name} requires exactly one of data, url, file_id, or path.")


def _coerce_message(value: dict[str, Any] | LMMessage) -> LMMessage:
    if isinstance(value, LMMessage):
        return value
    return LMMessage(**value)


def _messages_from_items(items: tuple[Any, ...], *, prompt: str | None = None) -> tuple[list[LMMessage], list[Any]]:
    if prompt is not None:
        items = (prompt, *items)
    if not items:
        items = ("",)

    if len(items) == 1 and _is_message_sequence(items[0]):
        items = tuple(items[0])

    if all(isinstance(item, LMMessage) or isinstance(item, LMResponse) for item in items):
        messages: list[LMMessage] = []
        for item in items:
            if isinstance(item, LMMessage):
                messages.append(item)
            else:
                messages.extend(_messages_from_response(item))
        return messages, []

    parts: list[LMPart] = []
    tools: list[Any] = []
    for item in items:
        if _is_dspy_tool(item):
            tools.append(item)
        else:
            parts.append(_coerce_part(item))
    return [LMMessage(role="user", parts=parts)], tools


def _messages_from_response(response: LMResponse) -> list[LMMessage]:
    return [LMMessage(role="assistant", parts=output.parts) for output in response.outputs]


def _is_message_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, LMMessage) or isinstance(item, LMResponse) for item in value
    )


def _coerce_part(value: Any) -> LMPart:
    if isinstance(
        value,
        (
            LMTextPart,
            LMImagePart,
            LMAudioPart,
            LMVideoPart,
            LMDocumentPart,
            LMBinaryPart,
            LMToolCallPart,
            LMToolResultPart,
            LMThinkingPart,
            LMCitationPart,
            LMRefusalPart,
        ),
    ):
        return value
    if isinstance(value, str):
        return LMTextPart(text=value)
    if isinstance(value, dict) and "type" in value:
        return pydantic.TypeAdapter(LMPart).validate_python(value)
    if _is_dspy_image(value):
        return _image_to_part(value)
    if _is_dspy_audio(value):
        return LMAudioPart(data=value.data, media_type=f"audio/{value.audio_format}")
    if _is_dspy_file(value):
        return _file_to_part(value)
    if _is_dspy_reasoning(value):
        return LMThinkingPart(text=value.content)
    if _is_dspy_tool_call(value):
        return LMToolCallPart(id=getattr(value, "id", None), name=value.name, args=value.args)
    raise TypeError(f"Cannot convert {type(value)!r} to an LMPart.")


def _parts_from_openai_content(content: Any) -> list[LMPart]:
    if isinstance(content, str):
        return [LMTextPart(text=content)]
    if not isinstance(content, list):
        return [_coerce_part(content)]

    parts = []
    for item in content:
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type == "text":
            parts.append(LMTextPart(text=item.get("text", "")))
        elif item_type == "image_url":
            url = item.get("image_url", {}).get("url", "")
            parts.append(_image_source_to_part(url))
        elif item_type == "input_audio":
            audio = item.get("input_audio", {})
            parts.append(LMAudioPart(data=audio.get("data"), media_type=f"audio/{audio.get('format', 'wav')}"))
        elif item_type == "file":
            parts.append(_binary_dict_to_part(item.get("file", {})))
        elif item_type == "document":
            parts.append(_document_dict_to_part(item))
        elif item_type == "video":
            video = item.get("video", {})
            parts.append(_media_dict_to_video_part(video))
        else:
            parts.append(_coerce_part(item))
    return parts


def _image_to_part(image: Any) -> LMImagePart:
    return _image_source_to_part(image.url)


def _image_source_to_part(source: str) -> LMImagePart:
    if source.startswith("data:"):
        media_type, data = _split_data_uri(source)
        return LMImagePart(data=data, media_type=media_type)
    media_type = mimetypes.guess_type(urlparse(source).path)[0] or "image/png"
    return LMImagePart(url=source, media_type=media_type)


def _file_to_part(file: Any) -> LMBinaryPart:
    if file.file_data is not None:
        media_type, data = _split_data_uri(file.file_data)
        return LMBinaryPart(data=data, media_type=media_type, filename=file.filename)
    if file.file_id is not None:
        return LMBinaryPart(file_id=file.file_id, filename=file.filename)
    raise ValueError("File must have file_data or file_id.")


def _binary_dict_to_part(file: dict[str, Any]) -> LMBinaryPart:
    if file.get("file_data") is not None:
        media_type, data = _split_data_uri(file["file_data"])
        return LMBinaryPart(data=data, media_type=media_type, filename=file.get("filename"))
    if file.get("data") is not None:
        media_type, data = _split_data_uri(file["data"])
        return LMBinaryPart(data=data, media_type=media_type, filename=file.get("filename"))
    if file.get("file_id") is not None:
        return LMBinaryPart(file_id=file["file_id"], filename=file.get("filename"))
    raise ValueError("Binary content block requires data, file_data, or file_id.")


def _document_dict_to_part(item: dict[str, Any]) -> LMDocumentPart:
    source = item.get("source")
    if isinstance(source, dict):
        return LMDocumentPart(
            source=source,
            citations=item.get("citations") or {},
            title=item.get("title"),
            context=item.get("context"),
        )
    if isinstance(source, str):
        media_type, data = _split_data_uri(source)
        return LMDocumentPart(data=data, media_type=media_type, title=item.get("title"), context=item.get("context"))
    raise ValueError("Document content block requires source.")


def _media_dict_to_video_part(video: dict[str, Any]) -> LMVideoPart:
    if video.get("data") is not None:
        media_type, data = _split_data_uri(video["data"])
        return LMVideoPart(data=data, media_type=media_type)
    if video.get("url") is not None:
        return LMVideoPart(url=video["url"], media_type=video.get("media_type") or "video/mp4")
    if video.get("file_id") is not None:
        return LMVideoPart(file_id=video["file_id"], media_type=video.get("media_type") or "video/mp4")
    raise ValueError("Video content block requires data, url, or file_id.")


def _split_data_uri(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        return "application/octet-stream", value
    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type, data


def _coerce_tool_spec(tool: Any) -> LMToolSpec:
    if isinstance(tool, LMToolSpec):
        return tool
    if hasattr(tool, "to_lm_tool_spec"):
        return tool.to_lm_tool_spec()
    if isinstance(tool, dict):
        if "function" in tool:
            function = tool["function"]
            return LMToolSpec(
                name=function.get("name"),
                description=function.get("description"),
                parameters=function.get("parameters", {}),
            )
        return LMToolSpec(**tool)
    raise TypeError(f"Cannot convert {type(tool)!r} to LMToolSpec.")


def _is_dspy_image(value: Any) -> bool:
    return value.__class__.__name__ == "Image" and hasattr(value, "url")


def _is_dspy_audio(value: Any) -> bool:
    return value.__class__.__name__ == "Audio" and hasattr(value, "data") and hasattr(value, "audio_format")


def _is_dspy_file(value: Any) -> bool:
    return value.__class__.__name__ == "File" and any(hasattr(value, attr) for attr in ("file_data", "file_id"))


def _is_dspy_reasoning(value: Any) -> bool:
    return value.__class__.__name__ == "Reasoning" and hasattr(value, "content")


def _is_dspy_tool(value: Any) -> bool:
    return value.__class__.__name__ == "Tool" and hasattr(value, "func")


def _is_dspy_tool_call(value: Any) -> bool:
    return (
        value.__class__.__name__ in {"ToolCall", "LMToolCallPart"} and hasattr(value, "name") and hasattr(value, "args")
    )


def _requires_output_dict(output: LMOutput) -> bool:
    return bool(
        output.logprobs is not None or output.reasoning_content is not None or output.tool_calls or output.citations
    )


def _part_to_value(part: LMPart) -> Any:
    if isinstance(part, LMTextPart):
        return part.text
    if isinstance(part, LMThinkingPart):
        return _reasoning_value(part.text)
    if isinstance(part, LMToolCallPart):
        return part
    if isinstance(part, LMRefusalPart):
        return part.text
    return part


def _reasoning_value(text: str) -> Any:
    try:
        from dspy.adapters.types.reasoning import Reasoning

        return Reasoning(text)
    except Exception:
        return LMThinkingPart(text=text)


def _tool_call_to_provider_dict(call: LMToolCallPart) -> dict[str, Any]:
    data = {
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.args),
        },
    }
    if call.id is not None:
        data["id"] = call.id
    return data


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
