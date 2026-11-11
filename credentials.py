"""
lm15.credentials — a credential is a closed sum, not a string.

spec/auth.md AUTH-2 (amended 2026-09-03, changes/2026-09-03-cloud-hosts.md):

    ApiKey         {"kind": "api_key", "value"}
    BearerToken    {"kind": "bearer_token", "value", "expires_at"?}
    AwsCredentials {"kind": "aws", "access_key_id", "secret_access_key",
                    "session_token"?, "expires_at"?}

A credential *provider* (the AUTH-2 zero-arg callable) returns one of these.
A plain string anywhere a credential is accepted reads as an ``ApiKey`` —
the shape every provider used before cloud hosts existed, kept so nothing
a user wrote breaks.  ``expires_at`` is an aware UTC datetime; absent
means non-expiring.

Secrecy (AUTH-5): reprs never show values; ``to_dict`` is the only way out
and callers that serialize a credential are the ones sending it on the
wire or writing a fixture on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, Union

__all__ = [
    "ApiKey",
    "AwsCredentials",
    "BearerToken",
    "CredentialKind",
    "CredentialValue",
    "CredentialProvider",
    "CredentialLike",
    "coerce_credential",
    "credential_from_dict",
    "credential_to_dict",
    "parse_rfc3339",
    "format_rfc3339",
]

CredentialKind = Literal["api_key", "bearer_token", "aws"]
CREDENTIAL_KINDS: frozenset[str] = frozenset({"api_key", "bearer_token", "aws"})

_EXPIRY_SKEW_SECONDS = 300  # AUTH-3: inside the skew window counts as expired.


def parse_rfc3339(value: str) -> datetime:
    """``2026-09-03T12:00:00Z`` (or an offset) → aware UTC datetime."""
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_rfc3339(value: datetime) -> str:
    """Aware datetime → ``YYYY-MM-DDTHH:MM:SSZ`` (whole seconds, UTC)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _expired(expires_at: datetime | None, now: datetime | None) -> bool:
    if expires_at is None:
        return False
    current = now if now is not None else datetime.now(timezone.utc)
    return (expires_at - current).total_seconds() <= _EXPIRY_SKEW_SECONDS


@dataclass(frozen=True, slots=True)
class ApiKey:
    value: str

    kind: Literal["api_key"] = field(default="api_key", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("ApiKey.value must be a non-empty string")

    def __repr__(self) -> str:  # AUTH-5
        return "ApiKey(<redacted>)"

    def is_expired(self, now: datetime | None = None) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class BearerToken:
    value: str
    expires_at: datetime | None = None

    kind: Literal["bearer_token"] = field(default="bearer_token", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("BearerToken.value must be a non-empty string")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("BearerToken.expires_at must be timezone-aware")

    def __repr__(self) -> str:  # AUTH-5
        tail = f", expires_at={format_rfc3339(self.expires_at)}" if self.expires_at else ""
        return f"BearerToken(<redacted>{tail})"

    def is_expired(self, now: datetime | None = None) -> bool:
        return _expired(self.expires_at, now)


@dataclass(frozen=True, slots=True)
class AwsCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None
    expires_at: datetime | None = None

    kind: Literal["aws"] = field(default="aws", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.access_key_id, str) or not self.access_key_id or not isinstance(self.secret_access_key, str) or not self.secret_access_key:
            raise ValueError("AwsCredentials needs non-empty string access_key_id and secret_access_key")
        if self.session_token is not None and (not isinstance(self.session_token, str) or not self.session_token):
            raise ValueError("AwsCredentials.session_token must be a non-empty string")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("AwsCredentials.expires_at must be timezone-aware")

    def __repr__(self) -> str:  # AUTH-5: the key id is not secret, the rest is.
        tail = f", expires_at={format_rfc3339(self.expires_at)}" if self.expires_at else ""
        return f"AwsCredentials(access_key_id={self.access_key_id!r}, <redacted>{tail})"

    def is_expired(self, now: datetime | None = None) -> bool:
        return _expired(self.expires_at, now)


CredentialValue = Union[ApiKey, BearerToken, AwsCredentials]
CredentialProvider = Callable[[], Union[str, CredentialValue]]
# What every ``api_key=`` argument accepts: a string, a value, or a provider.
CredentialLike = Union[str, CredentialValue, CredentialProvider]


def coerce_credential(value: str | CredentialValue) -> CredentialValue:
    """A string reads as an ``ApiKey``; a value passes through."""
    if isinstance(value, (ApiKey, BearerToken, AwsCredentials)):
        return value
    if isinstance(value, str):
        return ApiKey(value)
    raise TypeError(f"not a credential: {type(value).__name__}")


def credential_to_dict(value: CredentialValue) -> dict:
    """Canonical JSON (AUTH-2). Absent fields are omitted, never null."""
    if isinstance(value, ApiKey):
        return {"kind": "api_key", "value": value.value}
    if isinstance(value, BearerToken):
        out = {"kind": "bearer_token", "value": value.value}
        if value.expires_at is not None:
            out["expires_at"] = format_rfc3339(value.expires_at)
        return out
    if isinstance(value, AwsCredentials):
        out = {
            "kind": "aws",
            "access_key_id": value.access_key_id,
            "secret_access_key": value.secret_access_key,
        }
        if value.session_token is not None:
            out["session_token"] = value.session_token
        if value.expires_at is not None:
            out["expires_at"] = format_rfc3339(value.expires_at)
        return out
    raise TypeError(f"not a credential: {type(value).__name__}")


def credential_from_dict(data: dict) -> CredentialValue:
    kind = data.get("kind")
    expires = parse_rfc3339(str(data["expires_at"])) if data.get("expires_at") is not None else None
    if kind == "api_key":
        return ApiKey(data["value"])
    if kind == "bearer_token":
        return BearerToken(data["value"], expires_at=expires)
    if kind == "aws":
        token = data.get("session_token")
        return AwsCredentials(
            access_key_id=data["access_key_id"],
            secret_access_key=data["secret_access_key"],
            session_token=token,
            expires_at=expires,
        )
    raise ValueError("unknown credential kind")
