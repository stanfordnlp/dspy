"""Offline configuration-based backend selection, shared by planning and execution.

This decides which backend owns a model/client configuration. Native request
conversion still checks the actual message and generation-option representation
before execution. No provider client is constructed and no credential is invoked.
"""

import os
from dataclasses import dataclass, field

from dspy.clients.engines.lm15_engine import LM15Engine, timeouts_for
from dspy.lm15 import RouterConfig, UnsupportedFeatureError, _definitions

CLIENT_KEYS = {"api_key", "api_base", "base_url", "headers", "extra_headers", "timeout", "api_version",
               "azure_ad_token_provider", "organization", "project", "extra_query", "custom_llm_provider"}
NATIVE_CLIENT_KEYS = {"api_key", "api_base", "base_url", "timeout"}


@dataclass(frozen=True)
class BackendSelection:
    resolution: object = None
    clients: dict = field(default_factory=dict, repr=False)


def select_backend(lm, options=None, request=None):
    """Select using LM defaults plus call overrides, without inference I/O.

    Unknown providers and unsupported settings fail before inference; there
    is no second backend to reinterpret the destination or request.
    """
    spec = lm.engine
    if not isinstance(spec, str):
        raise TypeError("Custom engines supply their own capability contract")
    clients = {key: val for key, val in lm.kwargs.items() if key in CLIENT_KEYS}
    clients.update({key: val for key, val in (options or {}).items() if key in CLIENT_KEYS})
    providers = _definitions(getattr(lm, "_providers", ()))
    resolution = LM15Engine(RouterConfig(env={}, providers=providers), model_type=lm.model_type).resolve(lm.model)
    unsupported = sorted(key for key, value in clients.items() if key not in NATIVE_CLIENT_KEYS and value is not None)
    if unsupported:
        raise UnsupportedFeatureError(
            f"The native LM cannot carry these client settings: {unsupported}. "
            "Declare the endpoint with dspy.lm15.register_provider(...) or configure a custom engine."
        )
    clients = {key: value for key, value in clients.items() if key in NATIVE_CLIENT_KEYS}
    if request is not None:
        from dspy.clients.requests import passthrough_keys

        extra = passthrough_keys(request)
        if resolution.provider == "typesafe":
            # System One natively accepts structured instructions and criteria.
            extra -= {"questions"}
        if resolution.adapter == "AnthropicLM":
            # Messages owns the provider-shaped thinking object, including
            # budget and disable controls carried unchanged by the builder.
            extra -= {"thinking"}
        if extra:
            raise UnsupportedFeatureError(
                f"The native lm15 engine has no mapping for these provider options: {sorted(extra)}. "
                "Use a custom engine for provider-specific request handling."
            )
    # Preserve environment-configured gateways: never send their credentials
    # to the public provider endpoint. Explicit configuration wins over env.
    if not resolution.declared and not (clients.get("api_base") or clients.get("base_url")):
        prefix = (lm.model.split("/", 1)[0].split(":", 1)[0]
                  if "/" in lm.model or ":" in lm.model else resolution.provider.removesuffix("-chat")).upper()
        url = os.getenv(f"{prefix}_API_BASE") or os.getenv(f"{prefix}_BASE_URL")
        if url:
            clients["base_url"] = url
    if resolution.provider == "xai" and "api_key" not in clients and os.getenv("XAI_API_KEY"):
        clients["api_key"] = os.environ["XAI_API_KEY"]
    if "timeout" in clients:
        timeouts_for(clients["timeout"])
    return BackendSelection(resolution, clients)
