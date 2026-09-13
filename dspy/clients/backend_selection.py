"""Offline configuration-based backend selection, shared by planning and execution.

This decides which backend owns a model/client configuration. Native request
conversion still checks the actual message and generation-option representation
before execution. No provider client is constructed and no credential is invoked.
"""

import os
from dataclasses import dataclass, field

from dspy.clients.engines.lm15_engine import LM15Engine
from dspy.lm15 import RouterConfig, UnknownModelError, UnsupportedFeatureError

CLIENT_KEYS = {"api_key", "api_base", "base_url", "headers", "extra_headers", "timeout", "api_version",
               "azure_ad_token_provider", "organization", "project", "extra_query", "custom_llm_provider"}
NATIVE_CLIENT_KEYS = {"api_key", "api_base", "base_url", "timeout"}


@dataclass(frozen=True)
class BackendSelection:
    native: bool
    resolution: object = None
    clients: dict = field(default_factory=dict, repr=False)


def select_backend(lm, options=None):
    """Select using LM defaults plus call overrides, without inference I/O."""
    spec = lm._engine_spec
    if not isinstance(spec, str):
        raise TypeError("Custom engines supply their own capability contract")
    clients = {key: val for key, val in lm.kwargs.items() if key in CLIENT_KEYS}
    clients.update({key: val for key, val in (options or {}).items() if key in CLIENT_KEYS})
    native = spec != "litellm" and lm.model_type != "text"
    resolution = None
    if native:
        try:
            resolution = LM15Engine(RouterConfig(env={}), model_type=lm.model_type).resolve(lm.model)
            # Preserve legacy environment-configured gateways. A public native
            # endpoint must not receive credentials meant for that gateway.
            prefix = lm.model.split("/", 1)[0].upper()
            if spec == "auto" and any(os.getenv(name) for name in (f"{prefix}_API_BASE", f"{prefix}_BASE_URL")):
                native = False
            if spec == "auto" and resolution.provider == "xai" and "api_key" not in clients and os.getenv("XAI_API_KEY"):
                clients["api_key"] = os.environ["XAI_API_KEY"]
            if (set(clients) - NATIVE_CLIENT_KEYS) or (resolution.provider.startswith("azure") and
                                                       any(key in clients for key in ("api_base", "base_url"))):
                native = False
        except (UnknownModelError, UnsupportedFeatureError):
            if spec == "lm15":
                raise
            native = False
    if spec == "lm15" and not native:
        raise UnsupportedFeatureError("The requested client settings require the LiteLLM compatibility engine.")
    return BackendSelection(native, resolution, clients)
