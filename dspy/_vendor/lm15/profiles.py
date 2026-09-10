"""
lm15.profiles — Provider endpoint profiles and compatibility resolution.

ProviderProfile connects endpoint metadata, model metadata, and typed
compatibility policy. It is optional: provider adapters can still be used with
just api_key/base_url and Request(model="...").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .compat import (
    CompatProfile,
    OpenAIChatCompat,
    OpenAIResponsesCompat,
    ResolvedOpenAIResponsesCompat,
    merge_openai_responses_compat,
    resolve_openai_responses_compat as resolve_openai_responses_compat_partial,
)
from .models import (
    InferenceModelInfo,
    InferencePricing,
    ModelInfo,
    ModelOrigin,
    ModelRegistry,
)
from .types import JsonObject


# Re-export commonly used model/compat classes from lm15.profiles for ergonomic
# imports while keeping their implementation in compat.py/models.py.
__all__ = [
    "CompatProfile",
    "EndpointProfile",
    "InferenceModelInfo",
    "InferencePricing",
    "ModelInfo",
    "ModelOrigin",
    "ModelRegistry",
    "OpenAIChatCompat",
    "OpenAIResponsesCompat",
    "ProviderProfile",
    "ResolvedOpenAIResponsesCompat",
    "resolve_openai_responses_compat",
]


# ─── Validation helpers ──────────────────────────────────────────────


def _check_nonempty_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _check_json_object_or_none(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object or None")


# ─── Provider profiles ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EndpointProfile:
    name: str
    api_family: str
    base_url: str | None = None
    compat: CompatProfile | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_nonempty_text(self.name, "EndpointProfile.name")
        _check_nonempty_text(self.api_family, "EndpointProfile.api_family")
        if self.base_url is not None:
            _check_nonempty_text(self.base_url, "EndpointProfile.base_url")
        _check_json_object_or_none(self.extensions, "EndpointProfile.extensions")


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    provider: str
    endpoints: dict[str, EndpointProfile] = field(default_factory=dict)
    models: tuple[ModelInfo, ...] = ()
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_nonempty_text(self.provider, "ProviderProfile.provider")
        object.__setattr__(self, "endpoints", dict(self.endpoints))
        object.__setattr__(self, "models", tuple(self.models))
        for name, endpoint in self.endpoints.items():
            _check_nonempty_text(name, "ProviderProfile endpoint name")
            if not isinstance(endpoint, EndpointProfile):
                raise TypeError("ProviderProfile.endpoints values must be EndpointProfile")
        if not all(isinstance(m, ModelInfo) for m in self.models):
            raise TypeError("ProviderProfile.models must contain ModelInfo objects")
        _check_json_object_or_none(self.extensions, "ProviderProfile.extensions")

    @classmethod
    def inference(
        cls,
        *,
        provider: str,
        api_family: str,
        base_url: str | None = None,
        compat: CompatProfile | None = None,
        models: tuple[ModelInfo, ...] = (),
        extensions: JsonObject | None = None,
    ) -> "ProviderProfile":
        return cls(
            provider=provider,
            endpoints={
                "inference": EndpointProfile(
                    name="inference",
                    api_family=api_family,
                    base_url=base_url,
                    compat=compat,
                )
            },
            models=models,
            extensions=extensions,
        )

    def endpoint(self, name: str = "inference") -> EndpointProfile | None:
        return self.endpoints.get(name)

    def model(self, model_id: str) -> ModelInfo | None:
        for model in self.models:
            if model.id == model_id or model_id in model.aliases:
                return model
        return None


# ─── Compatibility resolution ────────────────────────────────────────


def openai_responses_compat_from_extensions(extensions: JsonObject | None) -> OpenAIResponsesCompat | None:
    """Read request-level OpenAI Responses compat from Config.extensions.

    Supported shapes:

        {"openai_responses_compat": {...}}
        {"openai_compat": {...}}                 # backwards-friendly alias
        {"compat": {"openai_responses": {...}}}
        {"compat": {"openai": {...}}}            # generic OpenAI alias

    This is an escape hatch. Normal configuration should use ProviderProfile.
    """
    if not extensions:
        return None

    raw: Any = extensions.get("openai_responses_compat")
    if raw is None:
        raw = extensions.get("openai_compat")
    if raw is None:
        compat = extensions.get("compat")
        if isinstance(compat, dict):
            raw = compat.get("openai_responses") or compat.get("openai")

    if not isinstance(raw, dict):
        return None

    allowed = {
        "developer_role",
        "max_output_tokens_field",
        "reasoning_format",
        "tool_result_name",
        "strict_tools",
        "cache_control",
        "commentary_phase",
        "edit_image_field",
        "builtin_tools",
        "tool_result_media",
        "routing",
        "extensions",
    }
    kwargs = {k: v for k, v in raw.items() if k in allowed}
    return OpenAIResponsesCompat(**kwargs)  # type: ignore[arg-type]


def resolve_openai_responses_compat(
    *,
    base_url: str,
    model: str,
    profile: ProviderProfile | None,
    request_extensions: JsonObject | None,
    base: OpenAIResponsesCompat | None = None,
) -> ResolvedOpenAIResponsesCompat:
    """Resolve effective OpenAI Responses compatibility policy.

    Layering:

        bound compat (``OpenAILM(compat=...)``), else the base URL default
        < endpoint compat
        < model compat
        < request extension override
    """
    partial = base if base is not None else _default_openai_responses_compat_for_base_url(base_url)

    endpoint = profile.endpoint("inference") if profile else None
    if isinstance(endpoint and endpoint.compat, OpenAIResponsesCompat):
        partial = merge_openai_responses_compat(partial, endpoint.compat)

    model_info = profile.model(model) if profile else None
    if isinstance(model_info and model_info.compat, OpenAIResponsesCompat):
        partial = merge_openai_responses_compat(partial, model_info.compat)

    partial = merge_openai_responses_compat(
        partial,
        openai_responses_compat_from_extensions(request_extensions),
    )
    return resolve_openai_responses_compat_partial(partial)


def _default_openai_responses_compat_for_base_url(base_url: str) -> OpenAIResponsesCompat:
    lower = base_url.lower()
    if "openrouter.ai" in lower:
        return OpenAIResponsesCompat.preset("openrouter")
    if "api.openai.com" in lower:
        return OpenAIResponsesCompat.preset("openai")
    if "api.meta.ai" in lower:
        return OpenAIResponsesCompat.preset("meta")
    # Preserve the current OpenAILM Responses serializer behavior for unknown
    # endpoints unless a profile/preset explicitly overrides it.
    return OpenAIResponsesCompat.preset("openai")
