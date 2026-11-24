"""
lm15.login.types — the public vocabulary of managed authentication.

spec/auth.md AUTH-12 (vocabulary), AUTH-13 (descriptors), AUTH-16 (the UI
boundary), AUTH-23 (model choices and selections), AUTH-24 (status).  Every
value here is secret-free by construction: a :class:`Connection` is
metadata about a saved credential, never the credential.  Objects that
carry secrets live in the flows and the store and never cross this
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "AuthUI",
    "Availability",
    "Connection",
    "ConnectionKind",
    "ConnectionStatus",
    "DeviceCodeNotice",
    "Flow",
    "InfoNotice",
    "LoginMethod",
    "ManualCodePrompt",
    "MethodField",
    "ModelChoice",
    "ModelSelection",
    "Notice",
    "ProgressNotice",
    "Prompt",
    "ProviderDescriptor",
    "SecretPrompt",
    "SelectOption",
    "SelectPrompt",
    "TextPrompt",
    "Usability",
    "Verification",
    "AuthUrlNotice",
]

ConnectionKind = Literal["account", "api_key", "cloud_identity", "local_server"]
Flow = Literal["authorization_code", "device_code", "form", "source_recipe"]
Availability = Literal["supported", "unavailable", "unverified"]
Usability = Literal["ready", "renewal_due", "needs_login", "indeterminate", "unknown"]


# ─── Descriptors (AUTH-13) ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MethodField:
    """One input a login method needs before it can start."""

    id: str
    label: str
    type: Literal["text", "secret", "select"] = "text"
    required: bool = True
    options: tuple["SelectOption", ...] = ()
    help: str | None = None


@dataclass(frozen=True, slots=True)
class LoginMethod:
    """A named way to establish a connection for a provider (AUTH-13.3).

    ``availability`` is the SDK's honest statement: ``supported`` means the
    implementation exists and has a recorded receipt; ``unverified`` means
    the code exists but no live receipt does (excluded from default pickers;
    explicit opt-in only); ``unavailable`` means it cannot run here, with
    ``reason`` saying why.  It never means "this account qualifies".
    """

    id: str
    label: str
    kind: ConnectionKind
    flow: Flow
    availability: Availability = "supported"
    reason: str | None = None
    fields: tuple[MethodField, ...] = ()
    delivery: tuple[str, ...] = ()  # loopback | manual | device
    subscription: bool = False       # backed by a provider subscription, per provider docs
    billing_note: str | None = None  # known provider policy; never an entitlement promise
    guidance: str | None = None

    @property
    def selectable(self) -> bool:
        return self.availability != "unavailable"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """A provider a manager can connect (AUTH-13.1).  ``id`` is the LM15
    route; ``service`` is a presentation group only."""

    id: str
    label: str
    service: str
    routes: tuple[str, ...]
    methods: tuple[LoginMethod, ...]
    docs_url: str | None = None
    console_url: str | None = None

    def method(self, method_id: str) -> LoginMethod:
        for method in self.methods:
            if method.id == method_id:
                return method
        raise KeyError(method_id)


# ─── Connections and status (AUTH-12, AUTH-24) ────────────────────────


@dataclass(frozen=True, slots=True)
class Connection:
    """Secret-free metadata for one saved credential in a scope."""

    id: str
    provider: str
    instance_id: str
    kind: ConnectionKind
    method_id: str
    routes: tuple[str, ...]
    label: str
    created_at: str
    identity_generation: str
    credential_revision: str
    settings: dict[str, str] = field(default_factory=dict)
    account_label: str | None = None  # untrusted display text (AUTH-12)

    def __repr__(self) -> str:
        return (
            f"Connection(id={self.id!r}, provider={self.provider!r}, kind={self.kind!r}, "
            f"method={self.method_id!r}, label={self.label!r})"
        )


@dataclass(frozen=True, slots=True)
class Verification:
    result: Literal["valid", "rejected", "unverified"]
    checked_at: str | None = None
    check: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ConnectionStatus:
    """AUTH-24: presence, usability and last verification are separate."""

    provider: str
    presence: Literal["saved", "absent"]
    usability: Usability
    connection: Connection | None = None
    expires_at: str | None = None  # RFC 3339, "never" or "unknown"
    logged_out: bool = False
    verification: Verification | None = None
    detail: str | None = None

    @property
    def ready(self) -> bool:
        return self.usability in ("ready", "renewal_due")


# ─── The UI boundary (AUTH-16) ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SelectOption:
    id: str
    label: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class TextPrompt:
    field_id: str
    label: str
    placeholder: str | None = None
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class SecretPrompt:
    field_id: str
    label: str
    placeholder: str | None = None
    type: Literal["secret"] = "secret"


@dataclass(frozen=True, slots=True)
class SelectPrompt:
    field_id: str
    label: str
    options: tuple[SelectOption, ...]
    type: Literal["select"] = "select"


@dataclass(frozen=True, slots=True)
class ManualCodePrompt:
    """Paste the redirect URL or code when the browser cannot reach the
    loopback listener.  Raced against the listener: the engine cancels
    whichever loses (AUTH-16)."""

    field_id: str
    label: str
    accepted: str = "the full redirect URL, or the code"
    type: Literal["manual_code"] = "manual_code"


Prompt = TextPrompt | SecretPrompt | SelectPrompt | ManualCodePrompt


@dataclass(frozen=True, slots=True)
class AuthUrlNotice:
    url: str
    instructions: str
    type: Literal["auth_url"] = "auth_url"


@dataclass(frozen=True, slots=True)
class DeviceCodeNotice:
    user_code: str
    verification_url: str
    expires_in_s: float
    interval_s: float
    type: Literal["device_code"] = "device_code"


@dataclass(frozen=True, slots=True)
class ProgressNotice:
    stage: str
    message: str
    type: Literal["progress"] = "progress"


@dataclass(frozen=True, slots=True)
class InfoNotice:
    message: str
    links: tuple[tuple[str, str], ...] = ()  # (label, url)
    type: Literal["info"] = "info"


Notice = AuthUrlNotice | DeviceCodeNotice | ProgressNotice | InfoNotice


@runtime_checkable
class AuthUI(Protocol):
    """What an application supplies to let a login talk to a person.

    ``prompt`` returns the answer (a ``SelectPrompt`` answer is the option
    id); it raises :class:`lm15.login.LoginCancelled` to cancel.  ``notify``
    shows a notice and returns.  A UI never opens a browser unless it is the
    application's explicit choice to do so (``TerminalUI(open_browser=True)``).
    """

    def prompt(self, prompt: Prompt) -> str: ...

    def notify(self, notice: Notice) -> None: ...

    def dismiss(self, prompt: Prompt) -> None:
        """Called when a displayed prompt became stale (the loopback callback
        arrived while a manual prompt was showing).  Optional."""


# ─── Model choices and selections (AUTH-23) ───────────────────────────


@dataclass(frozen=True, slots=True)
class ModelChoice:
    provider: str
    model: str
    connection_id: str
    source: Literal["bundled", "cached", "provider", "application", "manual"]
    label: str | None = None
    fetched_at: str | None = None
    capabilities: dict[str, Literal["supported", "unsupported", "unknown"]] = field(default_factory=dict)

    @property
    def routed(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """An exact route + model bound to one connection id and generation."""

    provider: str
    model: str
    connection_id: str
    identity_generation: str
    instance_id: str = "public"

    @property
    def routed(self) -> str:
        return f"{self.provider}:{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "model_id": self.model, "connection_id": self.connection_id,
            "identity_generation": self.identity_generation, "instance_id": self.instance_id,
        }
