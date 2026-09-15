"""
lm15.profiles — provider endpoint profiles (DEPRECATED) and the Responses
compat resolution that survives them.

``ProviderProfile`` / ``EndpointProfile`` / ``OpenAILM.from_profile`` /
``OpenAILM(profile=...)`` are deprecated as of 1.0.0rc2 and removed in
1.0.0 (contract ``changes/2026-09-11-job-handles-live-turns-profiles.md``
§ 3).  Everything they expressed has one home now:

- an endpoint's address and wire policy: ``OpenAILM(compat="ollama")``
  (a preset name supplies its own address) or ``compat=`` + ``base_url=``;
- a per-model policy: the chat dialect's ``OpenAIChatCompat.model_overrides``,
  or per request ``Config.extensions["openai_responses_compat"]``;
- a compat guessed from the base URL: nothing — say ``compat="openrouter"``.

What stays, in ``lm15.compat`` where it belongs: the partial-compat
semantics (``None`` inherits, ``"auto"`` is the adapter default),
``merge_openai_responses_compat``, and the request-level hatch
(:func:`openai_responses_compat_from_extensions`).  ``ModelInfo`` and
``ModelRegistry`` were never profiles; they live in ``lm15.models``.
"""

from __future__ import annotations

import warnings
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


_DEPRECATION = (
    "lm15.profiles.{name} is deprecated and will be removed in lm15 1.0.0: "
    "pass compat= (a preset name supplies its address) and base_url= to the "
    "adapter, model_overrides on OpenAIChatCompat, or "
    "Config.extensions['openai_responses_compat'] per request "
    "(lm15-contract changes/2026-09-11-job-handles-live-turns-profiles.md § 3)"
)


def _deprecated(name: str) -> None:
    warnings.warn(_DEPRECATION.format(name=name), DeprecationWarning, stacklevel=3)


@dataclass(frozen=True, slots=True)
class EndpointProfile:
    """DEPRECATED (1.0.0rc2; removed in 1.0.0). See the module docstring."""

    name: str
    api_family: str
    base_url: str | None = None
    compat: CompatProfile | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _deprecated("EndpointProfile")
        _check_nonempty_text(self.name, "EndpointProfile.name")
        _check_nonempty_text(self.api_family, "EndpointProfile.api_family")
        if self.base_url is not None:
            _check_nonempty_text(self.base_url, "EndpointProfile.base_url")
        _check_json_object_or_none(self.extensions, "EndpointProfile.extensions")


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """DEPRECATED (1.0.0rc2; removed in 1.0.0). See the module docstring."""

    provider: str
    endpoints: dict[str, EndpointProfile] = field(default_factory=dict)
    models: tuple[ModelInfo, ...] = ()
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _deprecated("ProviderProfile")
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

    The per-request hatch: the one place a Responses-dialect policy can
    differ per model after profiles are gone.  Normal configuration is
    ``OpenAILM(compat=...)``.
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
        < endpoint compat            (deprecated with profiles)
        < model compat               (deprecated with profiles)
        < request extension override

    The base-URL default is a guess about the server and is deprecated
    with the profiles: when it would pick anything but OpenAI's own
    policy it warns, naming the ``compat=`` spelling that says the same
    thing explicitly.
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
    guessed: str | None = None
    if "openrouter.ai" in lower:
        guessed = "openrouter"
    elif "api.meta.ai" in lower:
        guessed = "meta"
    if guessed is not None:
        warnings.warn(
            f"OpenAILM guessed compat={guessed!r} from base_url {base_url!r}; that guess is deprecated "
            f"and will be removed in lm15 1.0.0 — pass compat={guessed!r} explicitly (it also supplies "
            "the address)",
            DeprecationWarning,
            stacklevel=4,
        )
        return OpenAIResponsesCompat.preset(guessed)
    # OpenAI's own policy for OpenAI and for unknown endpoints, as before.
    return OpenAIResponsesCompat.preset("openai")
