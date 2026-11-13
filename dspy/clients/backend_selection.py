"""Offline configuration-based backend selection, shared by planning and execution.

This decides which backend owns a model/client configuration. Native request
conversion still checks the actual message and generation-option representation
before execution. No provider client is constructed and no credential is invoked.
"""

import os
from dataclasses import dataclass, field

from dspy._vendor.lm15.registry import lookup
from dspy.clients.engines.lm15_engine import LM15Engine, timeouts_for
from dspy.lm15 import RouterConfig, UnknownModelError, UnsupportedFeatureError, _definitions

CLIENT_KEYS = {"api_key", "api_base", "base_url", "headers", "extra_headers", "timeout", "api_version",
               "azure_ad_token_provider", "organization", "project", "extra_query", "custom_llm_provider"}
# timeout: since lm15 1.0.0rc2 the native engine takes it as RouterConfig
# timeouts (a number of seconds, or an httpx.Timeout); before, its presence
# sent the call to LiteLLM.
NATIVE_CLIENT_KEYS = {"api_key", "api_base", "base_url", "timeout"}


@dataclass(frozen=True)
class BackendSelection:
    native: bool
    resolution: object = None
    clients: dict = field(default_factory=dict, repr=False)
    model_type: str = "chat"


def compatibility_model(lm, selection):
    """Keep known compatibility spellings; normalize native OpenAI/Azure ids."""
    route = selection.resolution
    if lm.model_type == "auto" and route is not None and not route.declared:
        if route.provider in {"openai", "openai-chat", "azure", "azure-chat"}:
            return f"{route.provider.removesuffix('-chat')}/{route.model}"
        if selection.model_type == "responses":
            raise UnsupportedFeatureError(
                f"{lm.model!r} has no LiteLLM mapping for its native Responses route. "
                "Use native client settings or register the endpoint with its dialect and credentials."
            )
    return lm.model


def select_backend(lm, options=None, request=None):
    """Select using LM defaults plus call overrides, without inference I/O.

    A built Request may carry provider options lm15 has no verdict for in
    `Config.extensions`; only the LiteLLM compatibility engine can send them.
    """
    spec = lm.engine
    if not isinstance(spec, str):
        raise TypeError("Custom engines supply their own capability contract")
    prefix = lm.model.split("/", 1)[0].split(":", 1)[0].replace("_", "-")
    if prefix == "openai-decisions" and (spec == "litellm" or lm.model_type not in {"auto", "chat"}):
        raise UnsupportedFeatureError(
            "OpenAI Decisions has no LiteLLM or chat fallback; use engine='auto' or 'lm15' "
            "and model_type='auto'."
        )
    clients = {key: val for key, val in lm.kwargs.items() if key in CLIENT_KEYS}
    clients.update({key: val for key, val in (options or {}).items() if key in CLIENT_KEYS})
    native = spec != "litellm" and lm.model_type != "text"
    resolution = None
    providers = _definitions(getattr(lm, "_providers", ()))
    # A declared provider is resolved even under engine="litellm": its
    # LiteLLM route is built from the declaration (address, credential), not
    # from a model-string prefix LiteLLM may read as another service.
    if native or lm.model_type == "auto" or (providers and lm.model_type != "text"):
        try:
            resolution = LM15Engine(RouterConfig(env={}, providers=providers), model_type=lm.model_type).resolve(lm.model)
        except (UnknownModelError, UnsupportedFeatureError):
            if spec == "lm15":
                raise
            native = False
    model_type = lm.model_type
    if model_type == "auto":
        # Family selection is independent of backend selection: compatibility
        # must not send a Responses route to Chat Completions on fallback.
        definition = None
        if resolution is not None:
            definition = next((d for d in providers if d.id == resolution.provider), None) or lookup(resolution.provider)
        model_type = "responses" if definition is not None and definition.dialect == "openai-responses" else "chat"
    if native and request is not None:
        from dspy.clients.requests import passthrough_keys

        extra = passthrough_keys(request)
        if resolution.provider in {"typesafe", "openai-decisions"}:
            # Native decision backends accept structured instructions and criteria.
            extra -= {"questions"}
        if resolution.provider == "openrouter":
            # Preserve the native cross-model fallback policy.
            extra -= {"models", "route"}
        if resolution.adapter == "AnthropicLM":
            # Messages owns the provider-shaped thinking object, including
            # budget and disable controls carried unchanged by the builder.
            extra -= {"thinking"}
        if extra:
            if spec == "lm15":
                raise UnsupportedFeatureError(
                    f"The native lm15 engine has no mapping for these provider options: {sorted(extra)}. "
                    "Use engine='auto' or 'litellm' to send them through LiteLLM."
                )
            native = False
    if native and resolution is not None:
        try:
            # Gateways supported natively must not require the optional SDK.
            # Explicit configuration wins over environment defaults.
            if not resolution.declared and not (clients.get("api_base") or clients.get("base_url")):
                prefix = (lm.model.split("/", 1)[0].split(":", 1)[0]
                          if "/" in lm.model or ":" in lm.model else resolution.provider.removesuffix("-chat")).upper()
                url = os.getenv(f"{prefix}_API_BASE") or os.getenv(f"{prefix}_BASE_URL")
                if url:
                    clients["base_url"] = url
            if resolution.provider == "xai" and "api_key" not in clients and os.getenv("XAI_API_KEY"):
                clients["api_key"] = os.environ["XAI_API_KEY"]
            if any(key not in NATIVE_CLIENT_KEYS and value is not None for key, value in clients.items()):
                native = False
            else:
                clients = {key: value for key, value in clients.items() if key in NATIVE_CLIENT_KEYS}
            # A timeout the native engine cannot honor as written (a disabled
            # httpx component) is a client setting LiteLLM carries; a malformed
            # one stays a local TypeError/ValueError on every engine.
            if native and "timeout" in clients:
                timeouts_for(clients["timeout"])
        except (UnknownModelError, UnsupportedFeatureError):
            if spec == "lm15":
                raise
            native = False
    if spec == "litellm" and lm.model_type != "auto" and resolution is not None and not resolution.declared:
        resolution = None  # as before: the compatibility engine reads the model string itself
    if resolution is not None and resolution.provider == "openai-decisions" and not native:
        raise UnsupportedFeatureError(
            "OpenAI Decisions requires the native engine and native client settings; "
            "it has no LiteLLM or chat fallback."
        )
    if spec == "lm15" and not native:
        unsupported = sorted(key for key, value in clients.items() if key not in NATIVE_CLIENT_KEYS and value is not None)
        raise UnsupportedFeatureError(
            f"The native LM cannot carry these client settings: {unsupported}, or model_type={lm.model_type!r}. "
            "Use engine='auto' with dspy[litellm], or configure a custom engine."
        )
    return BackendSelection(native, resolution, clients, model_type)
