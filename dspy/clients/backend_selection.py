"""Offline configuration-based backend selection, shared by planning and execution.

This decides which backend owns a model/client configuration. Native request
conversion still checks the actual message and generation-option representation
before execution. No provider client is constructed and no credential is invoked.
"""

import os
from dataclasses import dataclass, field

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


def select_backend(lm, options=None):
    """Select using LM defaults plus call overrides, without inference I/O."""
    spec = lm._engine_spec
    if not isinstance(spec, str):
        raise TypeError("Custom engines supply their own capability contract")
    clients = {key: val for key, val in lm.kwargs.items() if key in CLIENT_KEYS}
    clients.update({key: val for key, val in (options or {}).items() if key in CLIENT_KEYS})
    native = spec != "litellm" and lm.model_type != "text"
    resolution = None
    providers = _definitions(getattr(lm, "_providers", ()))
    # A declared provider is resolved even under engine="litellm": its
    # LiteLLM route is built from the declaration (address, credential), not
    # from a model-string prefix LiteLLM may read as another service.
    if native or (providers and lm.model_type != "text"):
        try:
            resolution = LM15Engine(RouterConfig(env={}, providers=providers), model_type=lm.model_type).resolve(lm.model)
        except (UnknownModelError, UnsupportedFeatureError):
            if spec == "lm15":
                raise
            native = False
    if native and resolution is not None:
        try:
            # Preserve legacy environment-configured gateways. A public native
            # endpoint must not receive credentials meant for that gateway. A
            # declared provider states its address explicitly; an explicit
            # declaration beats an ambient variable.
            prefix = lm.model.split("/", 1)[0].upper()
            if spec == "auto" and not resolution.declared and any(
                os.getenv(name) for name in (f"{prefix}_API_BASE", f"{prefix}_BASE_URL")
            ):
                native = False
            if spec == "auto" and resolution.provider == "xai" and "api_key" not in clients and os.getenv("XAI_API_KEY"):
                clients["api_key"] = os.environ["XAI_API_KEY"]
            if (set(clients) - NATIVE_CLIENT_KEYS) or (resolution.provider.startswith("azure") and
                                                       any(key in clients for key in ("api_base", "base_url"))):
                native = False
            # A timeout the native engine cannot honor as written (a disabled
            # httpx component) is a client setting LiteLLM carries; a malformed
            # one stays a local TypeError/ValueError on every engine.
            if native and "timeout" in clients:
                timeouts_for(clients["timeout"])
        except (UnknownModelError, UnsupportedFeatureError):
            if spec == "lm15":
                raise
            native = False
    if spec == "litellm" and resolution is not None and not resolution.declared:
        resolution = None  # as before: the compatibility engine reads the model string itself
    if spec == "lm15" and not native:
        raise UnsupportedFeatureError("The requested client settings require the LiteLLM compatibility engine.")
    return BackendSelection(native, resolution, clients)
