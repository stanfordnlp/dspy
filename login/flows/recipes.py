"""
lm15.login.flows.recipes — connections that are recipes, not tokens.

Four kinds (AUTH-12 ``Connection.kind`` and AUTH-15's "recipe, not a
captured token"):

- ``api_key`` — a literal key the user typed or pasted (AUTH-16: literal
  text, no interpolation).  Stored as ``{"type": "api_key", "key": ...}``.
- ``env`` — *use the key from this environment variable*.  Stores the
  variable's name, never its value; read at request time from the process
  environment.  This is how ``connect()`` offers an ambient key as an
  explicit choice without copying it (R2/R3).
- ``external`` — *use the login another tool owns*: the Claude Code CLI
  (``~/.claude/.credentials.json``), the Codex CLI (``~/.codex/auth.json``),
  or the Pi agent's xAI store.  The file stays the owner; LM15 reads and
  renews through the legacy loaders exactly as the unmanaged router does
  (R1: existing access preserved, nothing copied).  Availability is
  ``supported`` because it is the path proven live since 2026-08-31.
- ``local_server`` — a keyless local engine (ollama, vllm, sglang): records
  the base URL and the preset's placeholder key.
"""

from __future__ import annotations

import os
from typing import Any

from ...credentials import ApiKey, BearerToken
from ..engine import LoginContext, LoginDenied
from ..types import LoginMethod, MethodField, ProviderDescriptor, SelectOption
from .base import LoginResult, Material, ProviderFlow, RequestAuth

EXTERNAL_SOURCES: dict[str, tuple[str, str]] = {
    # source id -> (provider route, human label)
    "claude-code-cli": ("claude-code", "your Claude Code login (~/.claude/.credentials.json)"),
    "codex-cli": ("openai-codex", "your Codex CLI login (~/.codex/auth.json)"),
    "pi-xai": ("xai", "your Pi agent xAI login (~/.pi/agent/auth.json)"),
}


def api_key_method(console_url: str | None = None) -> LoginMethod:
    return LoginMethod(
        id="api_key", label="Paste an API key", kind="api_key", flow="form",
        fields=(MethodField(id="key", label="API key", type="secret"),),
        guidance=f"Create one at {console_url}" if console_url else None,
        billing_note="Metered per token by the provider.",
    )


def env_method(env_keys: tuple[str, ...]) -> LoginMethod:
    return LoginMethod(
        id="env", label=f"Use the key in ${env_keys[0]} from the environment", kind="api_key", flow="source_recipe",
        fields=(MethodField(id="name", label="Environment variable", type="select",
                            options=tuple(SelectOption(k, f"${k}") for k in env_keys)),),
        billing_note="Metered per token by the provider; the variable's value is read at request time, never saved.",
    )


def external_method(source: str) -> LoginMethod:
    route, label = EXTERNAL_SOURCES[source]
    return LoginMethod(
        id=f"external:{source}", label=f"Use {label}", kind="account", flow="source_recipe",
        availability="supported", subscription=True,
        billing_note="Whatever that tool's login is entitled to; LM15 reads and renews it in place and copies nothing.",
        guidance="Sign in with that tool first if it says no credential is present.",
    )


class RecipeFlow(ProviderFlow):
    """One instance serves every provider: the material says what to do."""

    def __init__(self, provider: str, descriptor: ProviderDescriptor) -> None:
        self.provider = provider
        self.descriptor = descriptor

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        if method.id == "api_key":
            key = answers.get("key", "")
            if not isinstance(key, str) or not key.strip():
                raise LoginDenied("no API key was entered")
            return LoginResult(material={"type": "api_key", "key": key.strip()}, label=f"{self.provider} API key",
                               renewal="none")
        if method.id == "env":
            name = answers.get("name", "")
            if not name:
                raise LoginDenied("no environment variable was chosen")
            return LoginResult(material={"type": "env", "name": name}, label=f"{self.provider} key from ${name}",
                               renewal="recipe")
        if method.id.startswith("external:"):
            source = method.id.split(":", 1)[1]
            if source not in EXTERNAL_SOURCES:
                raise LoginDenied(f"unknown external source {source!r}")
            _probe_external(source)  # fail now, typed, if that tool has no login
            return LoginResult(material={"type": "external", "source": source},
                               label=f"{self.provider} via {EXTERNAL_SOURCES[source][1]}", renewal="external")
        if method.id == "cloud":
            named = answers.get("named", "")
            from ...features import NAMED_CREDENTIALS

            if named not in NAMED_CREDENTIALS:
                raise LoginDenied(f"choose one of {', '.join(NAMED_CREDENTIALS)}")
            return LoginResult(material={"type": "cloud", "named": named}, label=f"{self.provider} via {named} identity",
                               renewal="recipe")
        if method.id == "local":
            base_url = answers.get("base_url") or settings.get("base_url") or ""
            return LoginResult(material={"type": "local", "base_url": base_url, "key": answers.get("key") or "local"},
                               label=f"{self.provider} local server", renewal="none",
                               settings={"base_url": base_url} if base_url else {})
        raise ValueError(method.id)

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        # Recipes have nothing to renew: the external loaders renew in place
        # at request time; keys and env names do not expire.
        return LoginResult(material=dict(material), label=self.provider, renewal=material.get("type", "none"))

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        kind = material.get("type")
        if kind == "api_key":
            return RequestAuth(credential=ApiKey(material["key"]))
        if kind == "env":
            name = material["name"]
            value = os.environ.get(name, "")
            if not value:
                raise LoginDenied(f"${name} is not set in this process's environment")
            return RequestAuth(credential=ApiKey(value))
        if kind == "external":
            return _external_auth(material["source"])
        if kind == "local":
            return RequestAuth(credential=ApiKey(material.get("key") or "local"), base_url=material.get("base_url") or None)
        if kind == "cloud":
            return RequestAuth(credential=None, named=material["named"])
        raise LoginDenied(f"unknown connection material {kind!r}")

    def expiry(self, material: Material) -> int | str | None:
        kind = material.get("type")
        if kind in ("api_key", "env", "local", "cloud"):
            return "never"
        return None  # external: the owning tool knows


def _probe_external(source: str) -> None:
    from ... import auth as legacy

    route = EXTERNAL_SOURCES[source][0]
    if source == "claude-code-cli":
        legacy.load_claude_code_credential()
    elif source == "codex-cli":
        legacy.load_codex_cli_credential()
    elif source == "pi-xai":
        legacy.load_xai_credential(legacy.PI_AGENT_AUTH_PATH)
    else:
        raise LoginDenied(f"unknown external source for {route}")


def _external_auth(source: str) -> RequestAuth:
    """Resolve through the legacy loader: locked, double-checked renewal
    written back to the owning file (spec/auth.md AUTH-3/4)."""
    from ... import auth as legacy

    if source == "claude-code-cli":
        return RequestAuth(credential=BearerToken(legacy.get_claude_code_access_token()))
    if source == "codex-cli":
        credential = legacy.get_codex_cli_access_token()
        account = credential.account_id or legacy.extract_chatgpt_account_id(credential.access_token)
        return RequestAuth(credential=BearerToken(credential.access_token),
                           headers={"chatgpt-account-id": account} if account else {}, account_id=account)
    if source == "pi-xai":
        return RequestAuth(credential=BearerToken(legacy.get_xai_access_token(legacy.PI_AGENT_AUTH_PATH)))
    raise LoginDenied(f"unknown external source {source!r}")
