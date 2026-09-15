"""
lm15.features — Provider capability and endpoint declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

# How a provider's credential is obtained (spec/auth.md AUTH-1):
#
# - "key"   — the ordinary chain: explicit api_keys entry, then the declared
#             env vars in order, then (local-server presets only) the
#             preset's placeholder key.
# - "oauth" — self-resolving local OAuth only.  The adapter reads a local
#             credential file; the chain above never runs and a failed OAuth
#             load never falls back to an env var.
# - "oauth-unless-explicit" — an explicit api_keys entry wins; otherwise a
#             usable stored OAuth login wins; env vars are consulted only
#             when no usable login is stored.  Rationale: deliberate
#             in-process configuration always wins, but between two kinds of
#             stored state — a subscription login and an ambient env var —
#             the subscription wins because it spends no money per token
#             (lm15's durable constraint: normal inference must not
#             unexpectedly spend money).  Stated trade-off: with a login
#             stored, a set env key is silently ignored; pass the key
#             explicitly to force it.  Adapters declaring this policy MUST
#             expose ``has_stored_credential()`` — an offline, classmethod
#             probe the router calls to decide the chain.
#
# The router and doctor derive their behavior from this declaration; there
# is no parallel provider-name list to keep in sync.
# - "aws-chain" / "azure-chain" / "gcp-chain" — the cloud SDK's own default
#             resolution order (spec/auth.md AUTH-1, amended 2026-09-03),
#             rung by rung in ``lm15.cloud.chains``.  Rung 0 is the explicit
#             api_keys entry; the door's own key variable is rung 1; then
#             the SDK chain.  Same principal as boto3 / azure-identity /
#             google-auth on the same machine — anything else is a security
#             bug, not a convenience gap.
CredentialPolicy = Literal["key", "oauth", "oauth-unless-explicit", "aws-chain", "azure-chain", "gcp-chain"]
CREDENTIAL_POLICIES: frozenset[str] = frozenset(
    {"key", "oauth", "oauth-unless-explicit", "aws-chain", "azure-chain", "gcp-chain"}
)
_CLOUD_CHAINS: frozenset[str] = frozenset({"aws-chain", "azure-chain", "gcp-chain"})

# spec/auth.md AUTH-11: the mechanisms a cloud chain is made of.  A chain
# is data over these; a new kind is a spec change.
RungKind = Literal[
    "env", "ini-profile", "json-file", "subprocess", "http-metadata",
    "http-token-exchange", "sigv4-sts", "unsigned-sts", "jwt-rs256", "file-cache",
]
RUNG_KINDS: frozenset[str] = frozenset({
    "env", "ini-profile", "json-file", "subprocess", "http-metadata",
    "http-token-exchange", "sigv4-sts", "unsigned-sts", "jwt-rs256", "file-cache",
})

# spec/auth.md AUTH-7: what the doctor says about a rung.
AuthStepState = Literal["selected", "shadowed", "absent", "unprobed"]
AUTH_STEP_STATES: frozenset[str] = frozenset({"selected", "shadowed", "absent", "unprobed"})


@dataclass(frozen=True, slots=True)
class EndpointSupport:
    complete: bool = True
    stream: bool = True
    live: bool = False
    files: bool = False
    batches: bool = False
    images: bool = False
    speech: bool = False
    video: bool = False
    responses_api: bool = False
    models: bool = False
    caches: bool = False

    # Escape hatch for endpoint names not yet promoted to typed booleans.
    extra: frozenset[str] = field(default_factory=frozenset)

    def supports_endpoint(self, name: str) -> bool:
        if name in self.extra:
            return True
        return bool(getattr(self, name, False))


# How a credential travels (spec/vocabularies.md AuthScheme, AUTH-10).  A
# policy lists the schemes its door accepts, in preference order; the
# credential KIND picks one: an ApiKey goes out under the first header
# scheme, AwsCredentials under ``sigv4``.
AuthScheme = Literal["bearer", "x-api-key", "api-key", "query-key", "sigv4"]
AUTH_SCHEMES: frozenset[str] = frozenset({"bearer", "x-api-key", "api-key", "query-key", "sigv4"})
# The earlier two-value name, kept for callers that spelled the header.
AuthHeader = Literal["bearer", "x-api-key"]

StreamFraming = Literal["sse", "aws-event-stream"]
STREAM_FRAMINGS: frozenset[str] = frozenset({"sse", "aws-event-stream"})
ModelPlacement = Literal["body", "path"]
MODEL_PLACEMENTS: frozenset[str] = frozenset({"body", "path"})


@dataclass(frozen=True, slots=True)
class HostSetting:
    """One host setting: its name, the env variables consulted in order when
    the caller did not pass it, and its default (``None`` = required)."""

    name: str
    env: tuple[str, ...] = ()
    default: str | None = None


@dataclass(frozen=True, slots=True)
class HostSpec:
    """How a dialect reaches a cloud door (spec/auth.md AUTH-10 ``host``).

    ``base_url``           template over the settings: ``{region}``,
                           ``{project}``, ``{location}``, ``{location_host}``
                           (derived from ``location``), ``{resource}``.
    ``settings``           the settings this host needs, with env fallbacks.
    ``paths``              endpoint-path overrides keyed by the dialect's
                           endpoint name (``messages``, ``messages/stream``);
                           ``{model}`` is the request's model.  Absent means
                           the dialect's own path under ``base_url``.
    ``model_in``           ``body`` (default) or ``path`` — when ``path``,
                           the model field is removed from the payload.
    ``anthropic_version_in`` ``header`` (default) or ``body:<value>``.
    ``stream_framing``     ``sse`` or ``aws-event-stream``.
    ``required_headers``   ``(header name, setting name)`` pairs sent on every
                           request from the resolved settings.
    ``sigv4_service``      the SigV4 credential-scope service name.
    """

    base_url: str
    settings: tuple[HostSetting, ...] = ()
    paths: Mapping[str, str] = field(default_factory=dict)
    model_in: ModelPlacement = "body"
    anthropic_version_in: str = "header"
    stream_framing: StreamFraming = "sse"
    required_headers: tuple[tuple[str, str], ...] = ()
    sigv4_service: str | None = None

    def __post_init__(self) -> None:
        if self.model_in not in MODEL_PLACEMENTS:
            raise ValueError(f"HostSpec.model_in {self.model_in!r} not in {sorted(MODEL_PLACEMENTS)}")
        if self.stream_framing not in STREAM_FRAMINGS:
            raise ValueError(f"HostSpec.stream_framing {self.stream_framing!r} not in {sorted(STREAM_FRAMINGS)}")
        if self.anthropic_version_in != "header" and not self.anthropic_version_in.startswith("body:"):
            raise ValueError("HostSpec.anthropic_version_in is 'header' or 'body:<value>'")
        object.__setattr__(self, "settings", tuple(self.settings))
        object.__setattr__(self, "paths", dict(self.paths))
        object.__setattr__(self, "required_headers", tuple((str(k), str(v)) for k, v in self.required_headers))

    @property
    def setting_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.settings)


@dataclass(frozen=True, slots=True, init=False)
class AccessPolicy:
    """How an adapter reaches a backend. Pure data; ports copy it as a table.

    See ``lm15.access`` for the table of policies and the rationale
    (spec/auth.md AUTH-9).

    The constructor retains the original positional ``auth_header`` slot.
    Omitted (``None``), it leaves ``auth_scheme`` unchanged; explicitly supplied,
    it replaces the schemes with that single header, including in ``replace``.
    ``auth_scheme`` and ``host`` are keyword-only.

    ``provider``        canonical provider string (errors, routing, doctor).
    ``supports``        endpoint surfaces this access path carries; a
                        dialect that implements a surface still RAISES when
                        the policy does not carry it (a subscription token
                        has no files or batch).
    ``credential_policy`` spec/auth.md AUTH-1: key | oauth |
                        oauth-unless-explicit.
    ``auth_modes``, ``env_keys``, ``enterprise_variants``: as before
                        (support-matrix pinned).
    ``auth_scheme``     the schemes this door accepts, in preference order
                        (``bearer``, ``x-api-key``, ``api-key``,
                        ``query-key``, ``sigv4``); the credential kind
                        selects one.  ``auth_header`` is the first header
                        scheme, kept for the dialects that spell it.
    ``host``            the cloud door (``HostSpec``), or None for the
                        dialect's public API.
    ``headers``         static headers on every request, in order. A header
                        the dialect also sets is merged by the dialect's
                        stated rule (Anthropic joins ``anthropic-beta``).
    ``login_hint``      appended to auth errors when the credential is a
                        local login and there is no env var to set.
    ``backend``         dialect-consulted variant name; ``"api"`` is the
                        provider's public API.
    ``backend_options`` string knobs the backend variant needs (the Codex
                        models endpoint wants a ``client_version``).
    ``system_prefix``   text the backend requires first in the system
                        prompt / instructions (Claude Code, Codex).
    ``base_url``        this access path's default base URL, when it is
                        not the dialect's.
    """

    provider: str
    supports: EndpointSupport = field(default_factory=EndpointSupport)
    auth_modes: tuple[str, ...] = field(default_factory=tuple)
    enterprise_variants: tuple[str, ...] = field(default_factory=tuple)
    env_keys: tuple[str, ...] = field(default_factory=tuple)
    credential_policy: CredentialPolicy = "key"
    auth_scheme: tuple[AuthScheme, ...] = field(default=("bearer",), kw_only=True)
    headers: tuple[tuple[str, str], ...] = ()
    host: HostSpec | None = field(default=None, kw_only=True)
    login_hint: str | None = None
    backend: str = "api"
    backend_options: Mapping[str, str] = field(default_factory=dict)
    system_prefix: str | None = None
    base_url: str | None = None

    # Keep the original positional API, including pattern matching. The legacy
    # header is a constructor input/property, NOT a second stored authority.
    __match_args__ = (
        "provider", "supports", "auth_modes", "enterprise_variants", "env_keys",
        "credential_policy", "auth_header", "headers", "login_hint", "backend",
        "backend_options", "system_prefix", "base_url",
    )

    def __init__(
        self,
        provider: str,
        supports: EndpointSupport | None = None,
        auth_modes: tuple[str, ...] = (),
        enterprise_variants: tuple[str, ...] = (),
        env_keys: tuple[str, ...] = (),
        credential_policy: CredentialPolicy = "key",
        auth_header: AuthHeader | None = None,
        headers: tuple[tuple[str, str], ...] = (),
        login_hint: str | None = None,
        backend: str = "api",
        backend_options: Mapping[str, str] | None = None,
        system_prefix: str | None = None,
        base_url: str | None = None,
        *,
        auth_scheme: tuple[AuthScheme, ...] = ("bearer",),
        host: HostSpec | None = None,
    ) -> None:
        # None means the legacy spelling was omitted (the effective default
        # remains bearer). An explicit legacy header overrides the scheme input:
        # replace(policy, auth_header=...) also supplies the stored auth_scheme.
        # Keeping auth_header out of dataclass fields means ordinary replace and
        # with_headers never feed its lossy projection back into auth_scheme.
        if auth_header is not None:
            if auth_header not in ("bearer", "x-api-key"):
                raise ValueError(f"{provider}: unknown auth_header {auth_header!r}")
            auth_scheme = (auth_header,)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "supports", EndpointSupport() if supports is None else supports)
        object.__setattr__(self, "auth_modes", auth_modes)
        object.__setattr__(self, "enterprise_variants", enterprise_variants)
        object.__setattr__(self, "env_keys", env_keys)
        object.__setattr__(self, "credential_policy", credential_policy)
        object.__setattr__(self, "auth_scheme", auth_scheme)
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "login_hint", login_hint)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "backend_options", {} if backend_options is None else backend_options)
        object.__setattr__(self, "system_prefix", system_prefix)
        object.__setattr__(self, "base_url", base_url)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("AccessPolicy.provider must be non-empty")
        if self.credential_policy == "oauth" and self.env_keys:
            # AUTH-1: an oauth manifest declares no environment keys.
            raise ValueError(f"{self.provider}: an 'oauth' access policy declares no env_keys")
        if self.credential_policy not in CREDENTIAL_POLICIES:
            raise ValueError(f"{self.provider}: unknown credential_policy {self.credential_policy!r}")
        schemes = (self.auth_scheme,) if isinstance(self.auth_scheme, str) else tuple(self.auth_scheme)
        if not schemes:
            raise ValueError(f"{self.provider}: auth_scheme must name at least one scheme")
        for scheme in schemes:
            if scheme not in AUTH_SCHEMES:
                raise ValueError(f"{self.provider}: unknown auth_scheme {scheme!r}")
        if "sigv4" in schemes and (self.host is None or self.host.sigv4_service is None):
            raise ValueError(f"{self.provider}: sigv4 needs a host with sigv4_service")
        if self.credential_policy in _CLOUD_CHAINS and self.host is None and self.provider != "vertex-express":
            raise ValueError(f"{self.provider}: a cloud chain policy needs a host")
        object.__setattr__(self, "auth_scheme", schemes)
        object.__setattr__(self, "auth_modes", tuple(self.auth_modes))
        object.__setattr__(self, "env_keys", tuple(self.env_keys))
        object.__setattr__(self, "enterprise_variants", tuple(self.enterprise_variants))
        object.__setattr__(self, "headers", tuple((str(k), str(v)) for k, v in self.headers))
        object.__setattr__(self, "backend_options", dict(self.backend_options))

    @property
    def auth_header(self) -> AuthHeader:
        """The first header-carrying scheme (``bearer`` or ``x-api-key``);
        the two-value spelling the dialects consult for an ``ApiKey``.
        ``api-key`` reports as ``x-api-key`` here — dialects that can send
        either read ``auth_scheme`` directly."""
        for scheme in self.auth_scheme:
            if scheme == "bearer":
                return "bearer"
            if scheme in ("x-api-key", "api-key"):
                return "x-api-key"
        return "bearer"

    @property
    def cloud_chain(self) -> bool:
        return self.credential_policy in _CLOUD_CHAINS

    def with_headers(self, headers: Mapping[str, str]) -> "AccessPolicy":
        """A copy with these static headers replaced or appended (names compared case-insensitively)."""
        lowered = {k.lower() for k in headers}
        kept = tuple((k, v) for k, v in self.headers if k.lower() not in lowered)
        return replace(self, headers=kept + tuple(headers.items()))


# The earlier name: an adapter's manifest is its access policy.
ProviderManifest = AccessPolicy
