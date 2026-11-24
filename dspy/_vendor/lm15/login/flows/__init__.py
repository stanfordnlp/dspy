"""
lm15.login.flows — the provider table the manager consults.

One :class:`ProviderFlow` per provider LM15 can connect.  Account flows are
hand-written per provider (their protocols differ in ways a generic OAuth
machine would hide badly — D-3 in the decision log); key, env, external and
local recipes are generated from the provider registry so every route the
router knows can be connected the same way.

``descriptor(provider)`` and ``flow(provider)`` are the only entry points.
Descriptors are built once, lazily, from data; nothing here reads a
credential file, the network or the environment (AUTH-13.2).
"""

from __future__ import annotations

from functools import lru_cache

from ...registry import PROVIDERS, canonical_provider
from ..types import LoginMethod, MethodField, ProviderDescriptor, SelectOption
from .base import LoginResult, Material, ProviderFlow, RequestAuth
from .recipes import EXTERNAL_SOURCES, RecipeFlow, api_key_method, env_method, external_method

__all__ = ["ProviderFlow", "LoginResult", "RequestAuth", "Material", "descriptor", "flow", "provider_ids",
           "RADIUS_ID"]

RADIUS_ID = "radius"

# Providers whose account login LM15 implements natively (R11's inventory).
_ACCOUNT_FLOW_MODULES = {
    "xai": "xai", "claude-code": "claude", "openai-codex": "codex", "openrouter": "openrouter",
    "meta": "meta", "kimi-code": "kimi", "github-copilot": "copilot",
}

_SERVICE_LABELS = {
    "anthropic": "Anthropic", "claude-code": "Anthropic", "openai": "OpenAI", "openai-chat": "OpenAI",
    "openai-codex": "OpenAI", "gemini": "Google", "vertex": "Google Cloud", "vertex-anthropic": "Google Cloud",
    "vertex-express": "Google Cloud", "azure": "Microsoft Azure", "azure-chat": "Microsoft Azure",
    "azure-anthropic": "Microsoft Azure", "aws-anthropic": "AWS", "bedrock-anthropic": "AWS",
    "bedrock-chat": "AWS", "bedrock-mantle-chat": "AWS", "meta": "Meta", "meta-chat": "Meta",
    "meta-anthropic": "Meta", "moonshotai": "Moonshot AI", "moonshotai-anthropic": "Moonshot AI",
    "moonshotai-responses": "Moonshot AI", "kimi-code": "Moonshot AI", "deepseek": "DeepSeek",
    "deepseek-anthropic": "DeepSeek", "groq": "Groq", "openrouter": "OpenRouter", "xai": "xAI",
    "zai": "Z.AI", "typesafe": "TypeSafe", "ollama": "Local", "vllm": "Local", "sglang": "Local",
    "github-copilot": "GitHub", "deepinfra": "DeepInfra", "together": "Together AI", "fireworks": "Fireworks AI",
    "parasail": "Parasail",
}


def _account_flow(provider: str) -> ProviderFlow | None:
    module_name = _ACCOUNT_FLOW_MODULES.get(provider)
    if module_name is None:
        return None
    import importlib

    module = importlib.import_module(f".{module_name}", __package__)
    for name in dir(module):
        candidate = getattr(module, name)
        if isinstance(candidate, type) and issubclass(candidate, ProviderFlow) and candidate is not ProviderFlow:
            return candidate()
    raise RuntimeError(f"no flow class in lm15.login.flows.{module_name}")


def _recipe_methods(provider: str) -> tuple[LoginMethod, ...]:
    definition = PROVIDERS.get(provider)
    methods: list[LoginMethod] = []
    for source, (route, _label) in EXTERNAL_SOURCES.items():
        if route == provider:
            methods.append(external_method(source))
    if definition is None:
        return tuple(methods)
    access = definition.access
    if access.cloud_chain:
        from ...features import NAMED_CREDENTIALS

        methods.append(LoginMethod(
            id="cloud", label="Use a named cloud identity", kind="cloud_identity", flow="source_recipe",
            fields=(MethodField(id="named", label="Identity", type="select",
                                options=tuple(SelectOption(n, n) for n in NAMED_CREDENTIALS)),),
            billing_note="Billed to that cloud account.",
        ))
    if definition.placeholder_key is not None:
        methods.append(LoginMethod(
            id="local", label="Local server (no key needed)", kind="local_server", flow="source_recipe",
            fields=(MethodField(id="base_url", label="Server URL", required=False),),
        ))
        return tuple(methods)
    if access.credential_policy in ("key", "oauth-unless-explicit", "connection", "aws-chain", "azure-chain", "gcp-chain"):
        if access.credential_policy != "connection" or provider in _ACCOUNT_FLOW_MODULES:
            methods.append(api_key_method(definition.console_url))
        if access.env_keys:
            methods.append(env_method(tuple(access.env_keys)))
    return tuple(methods)


@lru_cache(maxsize=None)
def _table() -> dict[str, tuple[ProviderDescriptor, ProviderFlow, ProviderFlow | None]]:
    table: dict[str, tuple[ProviderDescriptor, ProviderFlow, ProviderFlow | None]] = {}
    ids = sorted(set(PROVIDERS) | set(_ACCOUNT_FLOW_MODULES))
    for provider in ids:
        account = _account_flow(provider)
        recipe_methods = _recipe_methods(provider)
        if account is not None:
            base = account.descriptor
            # The account flow's own api_key method (if any) is replaced by
            # the registry-derived recipe methods so every provider shares
            # one api_key/env implementation.
            own = tuple(m for m in base.methods if m.kind == "account" and not m.id.startswith("external:"))
            descriptor = ProviderDescriptor(
                id=base.id, label=base.label, service=base.service, routes=base.routes,
                methods=own + recipe_methods, docs_url=base.docs_url,
                console_url=base.console_url or (PROVIDERS[provider].console_url if provider in PROVIDERS else None),
            )
        else:
            definition = PROVIDERS[provider]
            descriptor = ProviderDescriptor(
                id=provider, label=provider, service=_SERVICE_LABELS.get(provider, provider),
                routes=(provider,), methods=recipe_methods, console_url=definition.console_url,
            )
        recipe = RecipeFlow(provider, descriptor)
        table[provider] = (descriptor, recipe, account)
    # Radius: named in R11's inventory; its gateway wire protocol is not in
    # lm15-python, so login alone would advertise a model connection that
    # cannot make a request (AUTH-26).  Listed, unavailable, honest.
    table[RADIUS_ID] = (
        ProviderDescriptor(
            id=RADIUS_ID, label="Radius", service="Radius", routes=(),
            methods=(LoginMethod(id="browser", label="Sign in with Radius", kind="account", flow="authorization_code",
                                 availability="unavailable",
                                 reason="Radius's model protocol is not implemented in lm15-python; login without "
                                        "inference would be a false 'supported' claim"),),
        ),
        RecipeFlow(RADIUS_ID, ProviderDescriptor(id=RADIUS_ID, label="Radius", service="Radius", routes=(), methods=())),
        None,
    )
    return table


def provider_ids() -> tuple[str, ...]:
    return tuple(sorted(_table()))


def descriptor(provider: str) -> ProviderDescriptor:
    canonical = canonical_provider(provider)
    try:
        return _table()[canonical][0]
    except KeyError:
        raise KeyError(provider) from None


def flow(provider: str, method_id: str) -> ProviderFlow:
    """The flow that implements ``method_id`` for ``provider``: the account
    flow for its own methods, the recipe flow for the generated ones."""
    canonical = canonical_provider(provider)
    _descriptor, recipe, account = _table()[canonical]
    if account is not None and any(m.id == method_id for m in account.descriptor.methods if m.kind == "account"):
        return account
    return recipe


def flow_for_material(provider: str, material: Material) -> ProviderFlow:
    canonical = canonical_provider(provider)
    _descriptor, recipe, account = _table()[canonical]
    kind = material.get("type")
    if kind in ("api_key", "env", "external", "local", "cloud") and not material.get("minted"):
        return recipe
    if kind == "api_key" and material.get("minted") and account is not None:
        return account
    if account is not None:
        return account
    return recipe
