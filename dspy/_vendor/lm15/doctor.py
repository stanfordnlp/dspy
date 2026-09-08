"""
lm15.doctor — explain how a provider's credential would resolve. No secrets.

:func:`explain_auth` answers "why is my key (not) being used?" without a
network call and without ever returning secret material. It walks the exact
chain the router's ``lm()`` walks — explicit ``api_keys`` entry, declared
environment variables in order, borrowed local CLI credentials for OAuth
providers, a local server's placeholder key — and reports the state of every
rung, including rungs that are set but shadowed by an earlier one.

Purity note, stated because it is a real trade-off: the router's
``resolve()`` records WHICH env var would be read and never touches values.
``explain_auth`` must go one step further and test env vars for presence
(``env.get(key)`` truthiness), so secret values do transit process memory.
They are never stored on the report, never included in ``describe()``, and
never part of any repr.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .auth import (
    CLAUDE_CODE_CREDENTIALS_PATH,
    CODEX_CLI_AUTH_PATH,
    LocalOAuthCredential,
    _load_xai_with_source,
    _xai_store_paths,
    read_claude_code_credential,
    read_codex_cli_credential,
)
from .errors import NotConfiguredError
from .providers import Credential
from .router import (
    ADAPTERS,
    RouterConfig,
    _api_keys_entry,
    _canonical_provider,
    _credential_policy,
    _declared_env_keys,
    _routable,
)

__all__ = ["AuthReport", "AuthStep", "explain_auth"]


@dataclass(frozen=True, slots=True)
class AuthStep:
    """One rung of the credential chain.

    ``state`` is one of:

    - ``"selected"`` — this rung supplies the credential;
    - ``"shadowed"`` — usable, but an earlier rung wins;
    - ``"absent"`` — nothing here;
    - ``"unprobed"`` — a network or subprocess rung of a cloud chain whose
      configuration is present; the offline doctor did not contact it
      (AUTH-7).  It runs before any later ``selected`` rung at request time
      and may win.

    ``kind`` is the language-neutral source identifier from the contract
    fixtures (lm15-contract/auth/resolution.json): ``"api_keys"``,
    ``"env:<VAR>"``, ``"placeholder"``, or ``"oauth-file"``. Conformance
    compares kinds, never display strings.

    ``detail`` is human text and carries no secret material by construction.
    """

    kind: str
    source: str
    detail: str
    state: str

    def describe(self) -> str:
        marker = {"selected": "=> ", "shadowed": " ~ ", "absent": " - ", "unprobed": " ? "}[self.state]
        return f"{marker}{self.source}: {self.detail}"


@dataclass(frozen=True, slots=True)
class AuthReport:
    provider: str
    steps: tuple[AuthStep, ...]
    configured: bool
    settings: tuple[tuple[str, str], ...] = ()  # resolved host settings (AUTH-7: printed by name and value)

    @property
    def selected(self) -> AuthStep | None:
        for step in self.steps:
            if step.state == "selected":
                return step
        return None

    def describe(self) -> str:
        lines = [f"auth for provider {self.provider!r}:"]
        lines += [f"  {step.describe()}" for step in self.steps]
        unprobed = [step for step in self.steps if step.state == "unprobed"]
        if self.configured and self.selected is not None:
            lines.append(f"  configured: yes — {self.selected.source}")
            if unprobed:
                lines.append(f"  note: {', '.join(s.source for s in unprobed)} run first at request time and may win")
        elif self.configured:
            lines.append(f"  configured: probably — {', '.join(s.source for s in unprobed)} (unprobed offline)")
        else:
            lines.append("  configured: no")
        for name, value in self.settings:
            lines.append(f"  setting {name}: {value}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def _expiry_detail(credential: LocalOAuthCredential) -> str:
    if credential.expires_at is None:
        return "no recorded expiry"
    remaining_ms = credential.expires_at - int(time.time() * 1000)
    if remaining_ms <= 0:
        suffix = "refresh token present" if credential.refresh_token else "NO refresh token"
        return f"expired, {suffix}"
    minutes = remaining_ms // 60_000
    hours, minutes = divmod(minutes, 60)
    span = f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"
    return f"fresh, expires in {span}"


def _usable_state(credential: LocalOAuthCredential, detail: str, shadowed: bool) -> str:
    if "expired" in detail and not credential.refresh_token:
        return "absent"
    return "shadowed" if shadowed else "selected"


def _oauth_step(provider: str, path_override: str | None) -> AuthStep:
    if provider == "claude-code":
        default_path, reader = CLAUDE_CODE_CREDENTIALS_PATH, read_claude_code_credential
    else:
        default_path, reader = CODEX_CLI_AUTH_PATH, read_codex_cli_credential
    path = Path(path_override).expanduser() if path_override else default_path
    source = f"local OAuth credential {path}"
    credential = reader(path)
    if credential is None:
        return AuthStep(kind="oauth-file", source=source, detail="missing or unreadable", state="absent")
    detail = _expiry_detail(credential)
    return AuthStep(kind="oauth-file", source=source, detail=detail, state=_usable_state(credential, detail, shadowed=False))


def _xai_oauth_step(path_override: str | None, shadowed: bool) -> AuthStep:
    """The stored xAI subscription login (lm15's own store, then the Pi
    agent store) — the middle rung of the oauth-unless-explicit chain:
    beaten only by an explicit api_keys entry, and itself beating env."""
    paths = (Path(path_override).expanduser(),) if path_override else _xai_store_paths()
    try:
        credential, path = _load_xai_with_source(path_override)
    except NotConfiguredError:
        checked = " or ".join(str(p) for p in paths)
        return AuthStep(kind="oauth-file", source=f"local OAuth credential {checked}", detail="missing or unreadable", state="absent")
    detail = _expiry_detail(credential)
    return AuthStep(
        kind="oauth-file",
        source=f"local OAuth credential {path}",
        detail=detail,
        state=_usable_state(credential, detail, shadowed=shadowed),
    )


def explain_auth(
    provider: str,
    *,
    env: Mapping[str, str] | None = None,
    api_keys: Mapping[str, Credential] | None = None,
    claude_credentials_path: str | None = None,
    codex_auth_path: str | None = None,
    xai_credentials_path: str | None = None,
    files: Mapping[str, str] | None = None,
    home: str | None = None,
    settings: Mapping[str, str] | None = None,
    config: RouterConfig | None = None,
) -> AuthReport:
    """Explain, rung by rung, how ``provider``'s credential resolves.

    Mirrors the router's construction chain exactly; divergence between this
    report and ``lm()`` behavior is a bug. ``env`` defaults to
    ``os.environ`` (pass a mapping for hermetic tests). Never returns or
    prints secret values, and performs no network I/O.

    ``config`` is the router's own ``RouterConfig`` (``router.config``): its
    ``env``, ``api_keys`` and the provider's ``settings`` entry are read, so
    the report describes the router that will send the request.  An
    explicit ``env=``/``api_keys=``/``settings=`` argument wins over the
    config's field.
    """
    import os

    canonical = _canonical_provider(provider)
    if config is not None:
        if env is None:
            env = config.env
        if api_keys is None:
            api_keys = config.api_keys
        if settings is None and config.settings is not None:
            settings = config.settings.get(canonical) or config.settings.get(provider)
    if not _routable(canonical, ADAPTERS):
        from .registry import PROVIDERS

        known = sorted(PROVIDERS)
        raise ValueError(f"Unknown provider {provider!r}. Known providers: {', '.join(known)}")

    policy = _credential_policy(canonical)
    if policy in ("aws-chain", "azure-chain", "gcp-chain") or (
        _bound_definition(canonical) is not None and _bound_definition(canonical).hosted
    ):
        return _explain_cloud(canonical, env=env, api_keys=api_keys, files=files, home=home, settings=settings)
    if policy == "oauth":
        override = claude_credentials_path if canonical == "claude-code" else codex_auth_path
        step = _oauth_step(canonical, override)
        return AuthReport(provider=canonical, steps=(step,), configured=step.state == "selected")

    config = RouterConfig(env=env, api_keys=api_keys)
    environment = env if env is not None else os.environ
    steps: list[AuthStep] = []
    selected = False

    _entry_value, has_entry = _api_keys_entry(config, canonical)
    if has_entry:
        steps.append(
            AuthStep(
                kind="api_keys",
                source="explicit api_keys entry",
                detail="provided (value never shown)",
                state="selected",
            )
        )
        selected = True
    else:
        steps.append(
            AuthStep(
                kind="api_keys",
                source="explicit api_keys entry",
                detail="not provided",
                state="absent",
            )
        )

    if policy == "oauth-unless-explicit":
        # The stored subscription login outranks env keys (AUTH-1): it
        # spends no money per token.  Only the explicit api_keys entry
        # above can shadow it.
        step = _xai_oauth_step(xai_credentials_path, shadowed=selected)
        steps.append(step)
        selected = selected or step.state == "selected"

    for key in _declared_env_keys(canonical, ADAPTERS):
        if environment.get(key):
            state = "shadowed" if selected else "selected"
            steps.append(
                AuthStep(kind=f"env:{key}", source=f"env ${key}", detail="set (value never shown)", state=state)
            )
            selected = True
        else:
            steps.append(
                AuthStep(kind=f"env:{key}", source=f"env ${key}", detail="not set", state="absent")
            )

    from .registry import PROVIDERS

    definition = PROVIDERS.get(canonical)
    if definition is not None and definition.placeholder_key is not None:
        state = "shadowed" if selected else "selected"
        steps.append(
            AuthStep(
                kind="placeholder",
                source="local-server placeholder key",
                detail=f"preset default for keyless {canonical} servers",
                state=state,
            )
        )
        selected = True

    return AuthReport(provider=canonical, steps=tuple(steps), configured=selected)


def _bound_definition(canonical: str):
    from .registry import PROVIDERS

    return PROVIDERS.get(canonical)


def _explain_cloud(
    canonical: str,
    *,
    env: Mapping[str, str] | None,
    api_keys: Mapping[str, Credential] | None,
    files: Mapping[str, str] | None,
    home: str | None,
    settings: Mapping[str, str] | None,
) -> AuthReport:
    """The cloud-chain walk (AUTH-1 aws/azure/gcp chains) through
    ``lm15.cloud.chains.explain``: offline, files from ``files`` when the
    harness materialized them, host settings resolved and printed."""
    import os

    from .cloud.chains import ChainContext, explain, profile_settings
    from .cloud.hosts import resolve_settings

    definition = _bound_definition(canonical)
    policy = definition.access
    environment = env if env is not None else os.environ
    config = RouterConfig(env=env, api_keys=api_keys)
    _value, has_entry = _api_keys_entry(config, canonical)
    resolved: dict[str, str] = {}
    setting_error: str | None = None
    ctx = ChainContext(
        env=environment,
        home=Path(home).expanduser() if home else (Path(environment["HOME"]) if environment.get("HOME") else Path.home()),
        files=files,
    )
    try:
        resolved = resolve_settings(policy.host, settings, environment, provider=canonical, profile=profile_settings(policy, ctx))
    except NotConfiguredError as exc:
        setting_error = str(exc).splitlines()[0]
    ctx.settings = resolved
    if policy.cloud_chain:
        steps, configured = explain(policy, ctx, explicit=has_entry)
        out = [AuthStep(kind=s.kind, source=s.source, detail=s.detail, state=s.state) for s in steps]
    else:
        # A hosted door with the ordinary key chain (vertex-express).
        out = [AuthStep("api_keys", "explicit api_keys entry",
                        "provided (value never shown)" if has_entry else "not provided",
                        "selected" if has_entry else "absent")]
        configured = has_entry
        for key in policy.env_keys:
            if environment.get(key):
                out.append(AuthStep(f"env:{key}", f"env ${key}", "set (value never shown)", "shadowed" if configured else "selected"))
                configured = True
            else:
                out.append(AuthStep(f"env:{key}", f"env ${key}", "not set", "absent"))
    shown = tuple(sorted(resolved.items()))
    if setting_error:
        shown = shown + (("error", setting_error),)
    return AuthReport(provider=canonical, steps=tuple(out), configured=configured, settings=shown)
