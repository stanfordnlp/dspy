"""
lm15.doctor — explain how a provider's credential would resolve. No secrets.

:func:`explain_auth` answers "why is my key (not) being used?" without a
network call and without ever returning secret material. It walks the exact
chain the router's ``lm()`` walks — explicit ``api_keys`` entry, declared
environment variables in order, borrowed local CLI credentials for OAuth
providers, a local server's placeholder key — and reports the state of every
rung, including rungs that are set but shadowed by an earlier one.

Purity note: both the router's lookup and ``explain_auth`` check
configured keys and environment presence, so secret values transit process
memory. Neither invokes credential providers during inspection. The doctor
also reads local credential files to report stored-login availability.
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
    Resolution,
    _api_keys_source,
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
    # AUTH-1 named credential (amended 2026-09-19): the name and what it
    # means on this cloud, when the config pins one; the steps are then
    # only the rungs it covers.
    named: str | None = None
    named_meaning: str | None = None
    # The base URL the door will send to, and where it came from
    # (``"base_urls"``, an env variable name, or ``"template"``).
    base_url: str | None = None
    base_url_source: str | None = None
    # Where each setting came from (AUTH-10 ``from`` vocabulary, amended
    # 2026-09-26): ``explicit``, ``env:<VAR>``, ``adc-env``,
    # ``gcloud-config``, ``adc-file``, ``metadata``, ``aws-profile``,
    # ``default``; ``unprobed:metadata`` when only the metadata server could
    # answer and the doctor does not use the network.
    setting_sources: tuple[tuple[str, str], ...] = ()

    @property
    def selected(self) -> AuthStep | None:
        for step in self.steps:
            if step.state == "selected":
                return step
        return None

    def describe(self) -> str:
        lines = [f"auth for provider {self.provider!r}:"]
        if self.named:
            lines.append(f'  named credential "{self.named}": {self.named_meaning} — the chain is not walked')
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
        origins = dict(self.setting_sources)
        for name, value in self.settings:
            origin = origins.get(name)
            shown = f"env ${origin[4:]}" if origin and origin.startswith("env:") else _SETTING_FROM.get(origin or "", origin)
            lines.append(f"  setting {name}: {value}" + (f" (from {shown})" if origin else ""))
        for name, origin in self.setting_sources:
            if origin.startswith("unprobed:"):  # "missing" is the error line below
                lines.append(f"  setting {name}: not found offline; {_SETTING_FROM.get(origin, origin)} is asked at request time")
        if self.base_url:
            origin = f" (from {self.base_url_source})" if self.base_url_source else ""
            lines.append(f"  base url: {self.base_url}{origin}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


_SETTING_FROM = {
    "explicit": "settings",
    "adc-env": "the GOOGLE_APPLICATION_CREDENTIALS file",
    "gcloud-config": "gcloud's active configuration",
    "adc-file": "the gcloud application default credentials file",
    "metadata": "the Google Cloud metadata server",
    "unprobed:metadata": "the Google Cloud metadata server",
    "aws-profile": "the active AWS profile",
    "default": "default",
}


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
    provider: str | Resolution,
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
    credential: str | None = None,
    base_url: str | None = None,
) -> AuthReport:
    """Explain, rung by rung, how ``provider``'s credential resolves.

    Accepts a provider name or a router ``Resolution``. A resolution
    supplies only the provider identity, not credentials: pass
    ``config=router.config`` to inspect that router's explicit settings.

    Mirrors the router's construction chain exactly; divergence between this
    report and ``lm()`` behavior is a bug. ``env`` defaults to
    ``os.environ`` (pass a mapping for hermetic tests). Never returns or
    prints secret values, and performs no network I/O.

    ``config`` is the router's own ``RouterConfig`` (``router.config``): its
    ``env``, ``api_keys``, ``credentials``, ``base_urls`` and the provider's
    ``settings`` entry are read, so the report describes the router that
    will send the request.  An explicit ``env=``/``api_keys=``/``settings=``
    /``credential=``/``base_url=`` argument wins over the config's field.

    On a cloud door the report also carries the base URL the door will
    send to and where it came from, and, under a named credential
    (``credential="platform"`` …), only the rungs that name covers.
    """
    import os

    if isinstance(provider, Resolution):
        provider = provider.provider
    if not isinstance(provider, str):
        raise TypeError("provider must be a provider name or Resolution")
    canonical = _canonical_provider(provider)
    if config is not None:
        if env is None:
            env = config.env
        if api_keys is None:
            api_keys = config.api_keys
        if settings is None and config.settings is not None:
            settings = config.settings.get(canonical) or config.settings.get(provider)
        if credential is None:
            from .router import _credentials_entry

            credential = _credentials_entry(config, canonical)
        if base_url is None:
            from .router import _base_url_entry

            base_url = _base_url_entry(config, canonical)
    if not _routable(canonical, ADAPTERS):
        from .registry import PROVIDERS

        known = sorted(PROVIDERS)
        raise ValueError(f"Unknown provider {provider!r}. Known providers: {', '.join(known)}")

    if config is not None and config.auth is not None:
        return _with_backend_settings(_explain_managed(canonical, config, api_keys=api_keys, env=env),
                                      settings, env)

    policy = _credential_policy(canonical)
    if policy in ("aws-chain", "azure-chain", "gcp-chain") or (
        _bound_definition(canonical) is not None and _bound_definition(canonical).hosted
    ):
        return _explain_cloud(canonical, env=env, api_keys=api_keys, files=files, home=home, settings=settings,
                              credential=credential, base_url=base_url)
    if credential is not None:
        raise NotConfiguredError(
            f"{canonical}: credential={credential!r} names a cloud identity, and this is not a cloud door",
            provider=canonical,
        )
    if policy == "oauth":
        override = claude_credentials_path if canonical == "claude-code" else codex_auth_path
        step = _oauth_step(canonical, override)
        return _with_backend_settings(AuthReport(provider=canonical, steps=(step,), configured=step.state == "selected"),
                                      settings, env)

    config = RouterConfig(env=env, api_keys=api_keys)
    environment = env if env is not None else os.environ
    steps: list[AuthStep] = []
    selected = False

    entry = _api_keys_source(config, canonical)
    if entry is not None:
        steps.append(
            AuthStep(
                kind="api_keys",
                source=_entry_source(canonical, entry),
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

    blocked = False
    if policy == "oauth-unless-explicit":
        # The stored subscription login outranks env keys (AUTH-1): it
        # spends no money per token.  Only the explicit api_keys entry
        # above can shadow it.  An unusable or signed-out login BLOCKS the
        # env keys (R3, 2026-09-22): they show as shadowed, and nothing is
        # selected.
        from .auth import xai_stored_state

        step = _xai_oauth_step(xai_credentials_path, shadowed=selected)
        state_word = xai_stored_state(xai_credentials_path)
        if state_word == "logged_out" and not selected:
            step = AuthStep(kind="oauth-file", source=step.source, detail="signed out (marker present)", state="absent")
            blocked = True
        elif state_word == "unusable" and not selected:
            blocked = True
        steps.append(step)
        selected = selected or step.state == "selected"

    for key in _declared_env_keys(canonical, ADAPTERS):
        if environment.get(key):
            state = "shadowed" if (selected or blocked) else "selected"
            detail = "set (value never shown)"
            if blocked and not selected:
                detail = "set, blocked by the failed/signed-out subscription (pass it explicitly to use it)"
            steps.append(AuthStep(kind=f"env:{key}", source=f"env ${key}", detail=detail, state=state))
            selected = selected or not blocked
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

    return _with_backend_settings(AuthReport(provider=canonical, steps=tuple(steps), configured=selected),
                                  settings, env)


def _with_backend_settings(report: AuthReport, settings: Mapping[str, str] | None,
                           env: Mapping[str, str] | None) -> AuthReport:
    """A door without a host prints its backend settings the way a cloud
    door prints its host settings (AUTH-7; AUTH-10 amended 2026-09-30):
    the Claude Code release the claude-code door claims, and where that
    came from — the value a model's minimum-version refusal is about."""
    import os
    from dataclasses import replace

    from .access import resolve_backend_settings

    definition = _bound_definition(report.provider)
    policy = (definition.access if definition is not None and definition.bound
              else getattr(ADAPTERS.get(report.provider), "manifest", None))
    if policy is None or policy.host is not None or (not policy.backend_settings and not settings):
        return report
    sources: dict[str, str] = {}
    values = resolve_backend_settings(policy, settings, env if env is not None else os.environ, sources=sources)
    return replace(report, settings=tuple(values.items()), setting_sources=tuple(sources.items()))


def _explain_managed(provider: str, config: RouterConfig, *, api_keys, env) -> AuthReport:
    """AUTH-15 mode B, rung by rung: the explicit entry, the named cloud
    identity, the scope's saved connection; environment keys are shown
    and marked not consulted.  Store reads only, no renewal (AUTH-7)."""
    import os

    from .router import _credentials_entry

    auth = config.auth
    walk = RouterConfig(env=env, api_keys=api_keys)
    environment = env if env is not None else os.environ
    steps: list[AuthStep] = []
    selected = False
    entry = _api_keys_source(walk, provider)
    if entry is not None:
        steps.append(AuthStep(kind="api_keys", source=_entry_source(provider, entry), detail="provided (value never shown)",
                              state="selected"))
        selected = True
    else:
        steps.append(AuthStep(kind="api_keys", source="explicit api_keys entry", detail="not provided", state="absent"))
    named = _credentials_entry(config, provider)
    if named is not None:
        steps.append(AuthStep(kind="named_cloud", source=f'named credential "{named}"', detail="explicit",
                              state="shadowed" if selected else "selected"))
        selected = True
    status = auth.status(provider)
    if status.connection is not None:
        detail = f"{status.connection.label} ({status.usability}" + (f", expires {status.expires_at}" if status.expires_at else "") + ")"
        state = "shadowed" if selected else ("selected" if status.ready else "absent")
        steps.append(AuthStep(kind="connection", source=f"saved connection {status.connection.id}", detail=detail, state=state))
        selected = selected or state == "selected"
    else:
        detail = "signed out (marker present)" if status.logged_out else "none saved in this scope"
        steps.append(AuthStep(kind="connection", source=f"saved connection in {auth.store.description}", detail=detail,
                              state="absent"))
    for key in _declared_env_keys(provider, ADAPTERS):
        if environment.get(key):
            steps.append(AuthStep(kind=f"env:{key}", source=f"env ${key}",
                                  detail="set, not consulted under a managed Auth (pass it explicitly to use it)",
                                  state="shadowed"))
        else:
            steps.append(AuthStep(kind=f"env:{key}", source=f"env ${key}", detail="not set", state="absent"))
    from .registry import PROVIDERS

    definition = PROVIDERS.get(provider)
    if definition is not None and definition.placeholder_key is not None and not status.logged_out:
        state = "shadowed" if selected else "selected"
        steps.append(AuthStep(kind="placeholder", source="local-server placeholder key",
                              detail=f"preset default for keyless {provider} servers", state=state))
        selected = True
    return AuthReport(provider=provider, steps=tuple(steps), configured=selected)


def _entry_source(provider: str, entry: str | None) -> str:
    source = "explicit api_keys entry"
    if entry is not None and _canonical_provider(entry) != provider:
        source += f" (via {entry!r}, shared env-key declarations)"
    return source


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
    credential: str | None = None,
    base_url: str | None = None,
) -> AuthReport:
    """The cloud-chain walk (AUTH-1 aws/azure/gcp chains) through
    ``lm15.cloud.chains.explain``: offline, files from ``files`` when the
    harness materialized them, host settings resolved and printed, the
    base URL and its origin named."""
    import os

    from .cloud.chains import ChainContext, explain, named_meaning, profile_settings
    from .cloud.hosts import endpoint_from_env, render_base_url, resolve_settings

    definition = _bound_definition(canonical)
    policy = definition.access
    environment = env if env is not None else os.environ
    config = RouterConfig(env=env, api_keys=api_keys, credentials={canonical: credential} if credential else None)
    if credential is not None:
        from .router import _check_named_credential

        _check_named_credential(config, canonical, credential, ADAPTERS)
    entry = _api_keys_source(config, canonical)
    has_entry = entry is not None
    resolved: dict[str, str] = {}
    setting_sources: dict[str, str] = {}
    setting_error: str | None = None
    ctx = ChainContext(
        env=environment,
        home=Path(home).expanduser() if home else (Path(environment["HOME"]) if environment.get("HOME") else Path.home()),
        files=files,
    )
    endpoint_source: str | None = None
    if base_url is not None:
        endpoint_source = "base_urls"
    elif policy.host is not None:
        base_url = endpoint_from_env(policy.host, environment)
        if base_url is not None:
            endpoint_source = next(var for var in policy.host.endpoint_env if (environment.get(var) or "").strip())
            endpoint_source = f"env ${endpoint_source}"
    try:
        problems: list[NotConfiguredError] = []
        resolved = resolve_settings(policy.host, settings, environment, provider=canonical,
                                    profile=profile_settings(policy, ctx), endpoint=base_url,
                                    sources=setting_sources, unprobed_ok=True, problems=problems)
        if problems:
            setting_error = str(problems[0]).splitlines()[0]
    except NotConfiguredError as exc:
        setting_error = str(exc).splitlines()[0]
    ctx.settings = resolved
    rendered: str | None = None
    pending = any(origin.startswith("unprobed:") for origin in setting_sources.values())
    if policy.host is not None and setting_error is None and not pending:  # nothing missing, nothing unprobed
        try:
            rendered = render_base_url(policy.host, resolved, base_url, provider=canonical)
        except NotConfiguredError as exc:
            setting_error = str(exc).splitlines()[0]
    if policy.cloud_chain:
        steps, configured = explain(policy, ctx, explicit=has_entry, named=credential)
        out = [AuthStep(kind=s.kind, source=_entry_source(canonical, entry) if s.kind == "api_keys" else s.source,
                        detail=s.detail, state=s.state) for s in steps]
    else:
        # A hosted door with the ordinary key chain (vertex-express).
        out = [AuthStep("api_keys", _entry_source(canonical, entry),
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
    return AuthReport(
        provider=canonical, steps=tuple(out), configured=configured, settings=shown,
        named=credential, named_meaning=named_meaning(policy, credential) if credential else None,
        base_url=rendered, base_url_source=(endpoint_source or "template") if rendered else None,
        setting_sources=tuple(sorted(setting_sources.items())),
    )
