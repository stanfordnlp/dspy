"""
lm15.auth — Local subscription credential helpers.

These helpers intentionally do not read ordinary provider API keys from the
environment.  They only support explicit local developer credentials created
by provider CLIs: Claude Code (``~/.claude/.credentials.json``), the OpenAI
Codex CLI (``~/.codex/auth.json``) — plus xAI subscription OAuth, which lm15
logs in itself (device-code flow) because xAI ships no CLI credential file
convention; see the xAI section at the bottom.

Failure behavior is typed and helpful, never a raw JSON traceback:

- missing / unreadable / malformed credential files raise
  :class:`lm15.errors.NotConfiguredError` telling the user which CLI login to
  run;
- an expired token that cannot be refreshed (no refresh token, or the refresh
  call fails) raises :class:`lm15.errors.AuthError` with the same re-login
  guidance;
- lock contention on a credential file raises
  :class:`lm15.auth.CredentialLockTimeout` (a ``TimeoutError``): it is a
  local, transient condition, not a provider error, so it deliberately does
  not wear the AuthError type.

Token material never appears in error messages or reprs.

Concurrency contract (lm15-contract spec/auth.md AUTH-3/AUTH-4, ratified
2026-08-31):

- refresh runs under a cross-process advisory lock, with a re-read of the
  file inside the lock (double-checked refresh) so a refresh that another
  process completed while we waited is used, not repeated — repeating it
  loses a rotated refresh token;
- all credential writes are atomic (temp file + rename) and private (0600);
- the lock is advisory and cooperative between lm15 processes. The Claude
  Code and Codex CLIs do not take it; the double-checked re-read is the
  mitigation for foreign writers, not a cure. The network refresh happens
  while holding the lock — one slow refresh can therefore stall other lm15
  processes for up to the lock timeout; the alternative (refresh outside the
  lock) double-spends rotated refresh tokens, which is worse.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._authlock import CredentialLockTimeout, hold_file_lock, write_private_json_atomic
from .errors import AuthError, NotConfiguredError, UnsupportedFeatureError

__all__ = [
    "CredentialLockTimeout",
    "LocalOAuthCredential",
    "XaiDeviceAuthorization",
    "get_claude_code_access_token",
    "get_codex_cli_access_token",
    "get_xai_access_token",
    "load_claude_code_credential",
    "load_codex_cli_credential",
    "load_xai_credential",
    "login",
    "login_xai",
    "poll_xai_device_login",
    "read_claude_code_credential",
    "read_codex_cli_credential",
    "read_xai_credential",
    "refresh_claude_code_credential",
    "refresh_codex_cli_credential",
    "refresh_xai_credential",
    "start_xai_device_login",
    "usable_xai_credential",
    "write_claude_code_credential",
    "write_codex_cli_credential",
    "write_xai_credential",
]

CLAUDE_CODE_CREDENTIALS_PATH = Path("~/.claude/.credentials.json").expanduser()
CLAUDE_CODE_CLIENT_ID = "9d1c250a-e61b-44d5-88ed-5944d1962f5e"
CLAUDE_CODE_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLAUDE_CODE_LOGIN_HINT = "Log in again: run `claude` and use /login (Claude subscription auth)"

CODEX_CLI_AUTH_PATH = Path("~/.codex/auth.json").expanduser()
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"
OPENAI_CODEX_LOGIN_HINT = "Log in again: run `codex login` (ChatGPT subscription auth)"

_REFRESH_SKEW_MS = 5 * 60 * 1000


@dataclass(frozen=True, slots=True)
class LocalOAuthCredential:
    """A locally stored OAuth credential.  Token fields are repr-suppressed."""

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: int | None = None
    account_id: str | None = None

    @property
    def expired(self) -> bool:
        return isinstance(self.expires_at, int) and int(time.time() * 1000) >= self.expires_at


def _not_configured(provider: str, message: str, hint: str) -> NotConfiguredError:
    return NotConfiguredError(message, provider=provider, credential_hint=hint)


def _read_json_file(path: Path, *, provider: str, hint: str) -> dict[str, Any]:
    """Read a credential file, raising typed errors instead of crashing."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise _not_configured(provider, f"No credentials file at {path}.", hint) from exc
    except OSError as exc:
        raise _not_configured(provider, f"Could not read credentials file at {path}: {exc}", hint) from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise _not_configured(provider, f"Credentials file at {path} is not valid JSON.", hint) from exc
    if not isinstance(data, dict):
        raise _not_configured(provider, f"Credentials file at {path} has an unexpected shape.", hint)
    return data


def _read_json_file_or_none(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_private_json(path: Path, data: dict[str, Any]) -> None:
    write_private_json_atomic(path, data)


def _base64url_json(segment: str) -> dict[str, Any]:
    padding = "=" * (-len(segment) % 4)
    decoded = base64.urlsafe_b64decode(segment + padding)
    data = json.loads(decoded.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT")
    return _base64url_json(parts[1])


def jwt_expires_at_ms(token: str) -> int | None:
    try:
        exp = decode_jwt_payload(token).get("exp")
    except Exception:
        return None
    if isinstance(exp, (int, float)):
        return int(exp * 1000) - _REFRESH_SKEW_MS
    return None


def extract_chatgpt_account_id(token: str) -> str | None:
    try:
        payload = decode_jwt_payload(token)
    except Exception:
        return None
    auth_claim = payload.get(OPENAI_CODEX_JWT_CLAIM_PATH)
    if isinstance(auth_claim, dict):
        account_id = auth_claim.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - provider token endpoint
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _post_form(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - provider token endpoint
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _coerce_path(path: str | os.PathLike[str] | None, default: Path) -> Path:
    return Path(path).expanduser() if path is not None else default


# ─── Claude Code (~/.claude/.credentials.json) ───────────────────────


def load_claude_code_credential(
    credentials_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential:
    """Load the Claude Code OAuth credential, raising typed errors."""
    path = _coerce_path(credentials_path, CLAUDE_CODE_CREDENTIALS_PATH)
    data = _read_json_file(path, provider="claude-code", hint=CLAUDE_CODE_LOGIN_HINT)
    raw = data.get("claudeAiOauth")
    if not isinstance(raw, dict):
        raise _not_configured(
            "claude-code",
            f"Credentials file at {path} has no claudeAiOauth section.",
            CLAUDE_CODE_LOGIN_HINT,
        )
    access = raw.get("accessToken")
    if not isinstance(access, str) or not access:
        raise _not_configured(
            "claude-code",
            f"Credentials file at {path} has no access token.",
            CLAUDE_CODE_LOGIN_HINT,
        )
    refresh = raw.get("refreshToken")
    expires = raw.get("expiresAt")
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh if isinstance(refresh, str) and refresh else None,
        expires_at=int(expires) if isinstance(expires, (int, float)) else None,
    )


def read_claude_code_credential(
    credentials_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential | None:
    """Optional-style loader: None when no usable credential exists."""
    try:
        return load_claude_code_credential(credentials_path)
    except NotConfiguredError:
        return None


def refresh_claude_code_credential(refresh_token: str) -> LocalOAuthCredential:
    payload = _post_json(
        CLAUDE_CODE_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": CLAUDE_CODE_CLIENT_ID,
            "refresh_token": refresh_token,
        },
    )
    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access, str) or not isinstance(refresh, str) or not isinstance(expires_in, (int, float)):
        raise RuntimeError("Claude Code token refresh response is missing required fields")
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh,
        expires_at=int(time.time() * 1000 + expires_in * 1000 - _REFRESH_SKEW_MS),
    )


def _write_claude_code_credential_unlocked(credential: LocalOAuthCredential, path: Path) -> None:
    data = _read_json_file_or_none(path) or {}
    raw = data.get("claudeAiOauth")
    current = raw if isinstance(raw, dict) else {}
    current["accessToken"] = credential.access_token
    if credential.refresh_token:
        current["refreshToken"] = credential.refresh_token
    if credential.expires_at is not None:
        current["expiresAt"] = credential.expires_at
    data["claudeAiOauth"] = current
    _write_private_json(path, data)


def write_claude_code_credential(
    credential: LocalOAuthCredential,
    credentials_path: str | os.PathLike[str] | None = None,
) -> None:
    path = _coerce_path(credentials_path, CLAUDE_CODE_CREDENTIALS_PATH)
    with hold_file_lock(path):
        _write_claude_code_credential_unlocked(credential, path)


def get_claude_code_access_token(
    credentials_path: str | os.PathLike[str] | None = None,
    *,
    refresh: bool = True,
) -> str:
    """Return a usable Claude Code access token, refreshing it if expired.

    Raises NotConfiguredError when no credential exists, AuthError when the
    credential is expired and cannot be refreshed.
    """
    credential = load_claude_code_credential(credentials_path)
    if not credential.expired:
        return credential.access_token
    if not refresh or not credential.refresh_token:
        raise AuthError(
            "Claude Code OAuth token is expired and no refresh token is available.",
            provider="claude-code",
            credential_hint=CLAUDE_CODE_LOGIN_HINT,
        )
    path = _coerce_path(credentials_path, CLAUDE_CODE_CREDENTIALS_PATH)
    with hold_file_lock(path):
        # Double-checked refresh: another process may have refreshed (and
        # rotated the refresh token) while we waited for the lock.
        credential = load_claude_code_credential(path)
        if not credential.expired:
            return credential.access_token
        if not credential.refresh_token:
            raise AuthError(
                "Claude Code OAuth token is expired and no refresh token is available.",
                provider="claude-code",
                credential_hint=CLAUDE_CODE_LOGIN_HINT,
            )
        try:
            refreshed = refresh_claude_code_credential(credential.refresh_token)
        except Exception as exc:
            raise AuthError(
                "Claude Code OAuth token is expired and the refresh attempt failed.",
                provider="claude-code",
                credential_hint=CLAUDE_CODE_LOGIN_HINT,
            ) from exc
        _write_claude_code_credential_unlocked(refreshed, path)
    return refreshed.access_token


# ─── OpenAI Codex CLI (~/.codex/auth.json) ───────────────────────────


def load_codex_cli_credential(
    auth_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential:
    """Load the Codex CLI OAuth credential, raising typed errors."""
    path = _coerce_path(auth_path, CODEX_CLI_AUTH_PATH)
    data = _read_json_file(path, provider="openai-codex", hint=OPENAI_CODEX_LOGIN_HINT)
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise _not_configured(
            "openai-codex",
            f"Credentials file at {path} has no tokens section.",
            OPENAI_CODEX_LOGIN_HINT,
        )
    access = tokens.get("access_token")
    if not isinstance(access, str) or not access:
        raise _not_configured(
            "openai-codex",
            f"Credentials file at {path} has no access token.",
            OPENAI_CODEX_LOGIN_HINT,
        )
    refresh = tokens.get("refresh_token")
    account_id = tokens.get("account_id") or extract_chatgpt_account_id(access)
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh if isinstance(refresh, str) and refresh else None,
        expires_at=jwt_expires_at_ms(access),
        account_id=account_id if isinstance(account_id, str) and account_id else None,
    )


def read_codex_cli_credential(
    auth_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential | None:
    """Optional-style loader: None when no usable credential exists."""
    try:
        return load_codex_cli_credential(auth_path)
    except NotConfiguredError:
        return None


def refresh_codex_cli_credential(refresh_token: str) -> LocalOAuthCredential:
    payload = _post_form(
        OPENAI_CODEX_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OPENAI_CODEX_CLIENT_ID,
        },
    )
    access = payload.get("access_token")
    refresh = payload.get("refresh_token") or refresh_token
    if not isinstance(access, str) or not isinstance(refresh, str):
        raise RuntimeError("Codex token refresh response is missing required fields")
    account_id = extract_chatgpt_account_id(access)
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh,
        expires_at=jwt_expires_at_ms(access),
        account_id=account_id,
    )


def write_codex_cli_credential(
    credential: LocalOAuthCredential,
    auth_path: str | os.PathLike[str] | None = None,
    *,
    id_token: str | None = None,
) -> None:
    path = _coerce_path(auth_path, CODEX_CLI_AUTH_PATH)
    with hold_file_lock(path):
        _write_codex_cli_credential_unlocked(credential, path, id_token=id_token)


def _write_codex_cli_credential_unlocked(
    credential: LocalOAuthCredential,
    path: Path,
    *,
    id_token: str | None = None,
) -> None:
    data = _read_json_file_or_none(path) or {}
    tokens = data.get("tokens")
    current = tokens if isinstance(tokens, dict) else {}
    current["access_token"] = credential.access_token
    if credential.refresh_token:
        current["refresh_token"] = credential.refresh_token
    if credential.account_id:
        current["account_id"] = credential.account_id
    if id_token is not None:
        current["id_token"] = id_token
    data["tokens"] = current
    data.setdefault("auth_mode", "chatgpt")
    data["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_private_json(path, data)


def get_codex_cli_access_token(
    auth_path: str | os.PathLike[str] | None = None,
    *,
    refresh: bool = True,
) -> LocalOAuthCredential:
    """Return a usable Codex CLI credential, refreshing it if expired.

    Raises NotConfiguredError when no credential exists, AuthError when the
    credential is expired and cannot be refreshed.
    """
    credential = load_codex_cli_credential(auth_path)
    if not credential.expired:
        return credential
    if not refresh or not credential.refresh_token:
        raise AuthError(
            "Codex CLI OAuth token is expired and no refresh token is available.",
            provider="openai-codex",
            credential_hint=OPENAI_CODEX_LOGIN_HINT,
        )
    path = _coerce_path(auth_path, CODEX_CLI_AUTH_PATH)
    with hold_file_lock(path):
        # Double-checked refresh: see the Claude Code sibling above.
        credential = load_codex_cli_credential(path)
        if not credential.expired:
            return credential
        if not credential.refresh_token:
            raise AuthError(
                "Codex CLI OAuth token is expired and no refresh token is available.",
                provider="openai-codex",
                credential_hint=OPENAI_CODEX_LOGIN_HINT,
            )
        try:
            refreshed = refresh_codex_cli_credential(credential.refresh_token)
        except Exception as exc:
            raise AuthError(
                "Codex CLI OAuth token is expired and the refresh attempt failed.",
                provider="openai-codex",
                credential_hint=OPENAI_CODEX_LOGIN_HINT,
            ) from exc
        original = _read_json_file_or_none(path) or {}
        tokens = original.get("tokens") if isinstance(original.get("tokens"), dict) else {}
        id_token = (
            tokens.get("id_token")
            if isinstance(tokens, dict) and isinstance(tokens.get("id_token"), str)
            else None
        )
        _write_codex_cli_credential_unlocked(refreshed, path, id_token=id_token)
    return refreshed


# ─── xAI Grok (device-code OAuth; lm15 store or Pi agent store) ──────
#
# Unlike Claude Code and Codex, xAI has no first-party CLI credential file
# convention lm15 can piggyback on.  lm15 therefore owns a login
# (:func:`login_xai`, RFC 8628 device-code flow) writing to lm15's own
# credential store, and additionally reads the Pi coding agent's store
# (``~/.pi/agent/auth.json``) when present — both files share the same
# ``{"xai": {"type": "oauth", "access", "refresh", "expires"}}`` schema.
# Refreshes write back to whichever file the credential came from: xAI
# rotates refresh tokens, so a refresh that is not persisted to its source
# bricks that source's login.

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_LOGIN_HINT = "Log in again: run lm15.auth.login_xai() (SuperGrok / X Premium subscription auth)"
PI_AGENT_AUTH_PATH = Path("~/.pi/agent/auth.json").expanduser()

_XAI_PROVIDER_KEY = "xai"
_XAI_DEFAULT_TOKEN_LIFETIME_S = 3600


def _xai_store_paths() -> tuple[Path, ...]:
    from .authkit import default_credentials_path

    return (default_credentials_path(), PI_AGENT_AUTH_PATH)


def _xai_entry_to_credential(entry: Any) -> LocalOAuthCredential | None:
    if not isinstance(entry, dict):
        return None
    access = entry.get("access")
    if not isinstance(access, str) or not access:
        return None
    refresh = entry.get("refresh")
    expires = entry.get("expires")
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh if isinstance(refresh, str) and refresh else None,
        expires_at=expires if isinstance(expires, int) and not isinstance(expires, bool) else None,
    )


def _xai_credential_to_entry(credential: LocalOAuthCredential, current: dict[str, Any] | None) -> dict[str, Any]:
    entry = dict(current or {})
    entry["type"] = "oauth"
    entry["access"] = credential.access_token
    if credential.refresh_token:
        entry["refresh"] = credential.refresh_token
    if credential.expires_at is not None:
        entry["expires"] = credential.expires_at
    return entry


def _load_xai_with_source(
    auth_path: str | os.PathLike[str] | None = None,
) -> tuple[LocalOAuthCredential, Path]:
    paths = (Path(auth_path).expanduser(),) if auth_path is not None else _xai_store_paths()
    for path in paths:
        data = _read_json_file_or_none(path)
        credential = _xai_entry_to_credential(data.get(_XAI_PROVIDER_KEY)) if data else None
        if credential is not None:
            return credential, path
    checked = ", ".join(str(p) for p in paths)
    raise _not_configured(
        "xai",
        f"No xAI OAuth credential found (checked: {checked}).",
        XAI_LOGIN_HINT,
    )


def load_xai_credential(
    auth_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential:
    """Load the stored xAI OAuth credential, raising typed errors."""
    credential, _ = _load_xai_with_source(auth_path)
    return credential


def read_xai_credential(
    auth_path: str | os.PathLike[str] | None = None,
) -> LocalOAuthCredential | None:
    """Optional-style loader: None when no usable credential exists."""
    try:
        return load_xai_credential(auth_path)
    except NotConfiguredError:
        return None


def usable_xai_credential(auth_path: str | os.PathLike[str] | None = None) -> bool:
    """True when a stored xAI subscription credential exists and is usable
    (fresh, or expired with a refresh token to refresh it at request time).

    Reads files only — never the network — so the router can consult it
    while staying offline.  This is the probe behind the
    ``oauth-unless-explicit`` credential policy (spec/auth.md AUTH-1): the
    stored subscription wins over ambient environment keys exactly when
    this returns True.
    """
    credential = read_xai_credential(auth_path)
    if credential is None:
        return False
    return not credential.expired or bool(credential.refresh_token)


def _xai_credential_from_token_response(
    payload: dict[str, Any],
    previous_refresh_token: str | None = None,
) -> LocalOAuthCredential:
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise RuntimeError("xAI token response is missing access_token")
    refresh = payload.get("refresh_token")
    if not isinstance(refresh, str) or not refresh:
        # xAI may omit refresh_token when it does not rotate it.
        refresh = previous_refresh_token
    expires_in = payload.get("expires_in")
    lifetime_s = (
        expires_in
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) and expires_in > 0
        else _XAI_DEFAULT_TOKEN_LIFETIME_S
    )
    return LocalOAuthCredential(
        access_token=access,
        refresh_token=refresh,
        expires_at=int(time.time() * 1000 + lifetime_s * 1000 - _REFRESH_SKEW_MS),
    )


def refresh_xai_credential(refresh_token: str) -> LocalOAuthCredential:
    payload = _post_form(
        XAI_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": XAI_CLIENT_ID,
            "refresh_token": refresh_token,
        },
    )
    return _xai_credential_from_token_response(payload, refresh_token)


def write_xai_credential(
    credential: LocalOAuthCredential,
    auth_path: str | os.PathLike[str] | None = None,
) -> None:
    """Write the credential into a store file (default: lm15's own store)."""
    from .authkit import CredentialFileStore, default_credentials_path

    store = CredentialFileStore(auth_path if auth_path is not None else default_credentials_path())
    store.mutate(_XAI_PROVIDER_KEY, lambda current: _xai_credential_to_entry(credential, current))


def get_xai_access_token(
    auth_path: str | os.PathLike[str] | None = None,
    *,
    refresh: bool = True,
) -> str:
    """Return a usable xAI access token, refreshing (and persisting) if expired.

    Raises NotConfiguredError when no credential exists, AuthError when the
    credential is expired and cannot be refreshed.  A refresh is written back
    to the file the credential came from (lm15 store or Pi agent store) under
    that file's lock, with a double-checked re-read: xAI rotates refresh
    tokens, so losing the write would break every consumer of that file.
    """
    from .authkit import CredentialFileStore

    credential, source = _load_xai_with_source(auth_path)
    if not credential.expired:
        return credential.access_token
    if not refresh or not credential.refresh_token:
        raise AuthError(
            "xAI OAuth token is expired and no refresh token is available.",
            provider="xai",
            credential_hint=XAI_LOGIN_HINT,
        )

    result: dict[str, str] = {}

    def _refresh_entry(current: dict[str, Any] | None) -> dict[str, Any] | None:
        # Double-checked refresh: another process may have refreshed (and
        # rotated the refresh token) while we waited for the lock.
        fresh = _xai_entry_to_credential(current)
        if fresh is not None and not fresh.expired:
            result["access"] = fresh.access_token
            return None
        refresh_token = (fresh.refresh_token if fresh else None) or credential.refresh_token
        if not refresh_token:
            raise AuthError(
                "xAI OAuth token is expired and no refresh token is available.",
                provider="xai",
                credential_hint=XAI_LOGIN_HINT,
            )
        try:
            refreshed = refresh_xai_credential(refresh_token)
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError(
                "xAI OAuth token is expired and the refresh attempt failed.",
                provider="xai",
                credential_hint=XAI_LOGIN_HINT,
            ) from exc
        result["access"] = refreshed.access_token
        return _xai_credential_to_entry(refreshed, current)

    CredentialFileStore(source).mutate(_XAI_PROVIDER_KEY, _refresh_entry)
    return result["access"]


# ─── xAI device-code login (RFC 8628) ────────────────────────────────


@dataclass(frozen=True, slots=True)
class XaiDeviceAuthorization:
    """One pending device authorization.  The device code is repr-suppressed."""

    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    interval_s: float
    expires_in_s: float
    device_code: str = field(repr=False, default="")


def _https_or_raise(raw: Any) -> str:
    # The verification URI is shown to (and often opened by) the user; refuse
    # anything a malicious response could turn into a local scheme launch.
    if isinstance(raw, str):
        try:
            parsed = urllib.parse.urlparse(raw)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.scheme == "https" and parsed.netloc:
            return raw
    raise AuthError("xAI device authorization returned an untrusted verification URI.", provider="xai")


def _parse_xai_device_authorization(payload: dict[str, Any]) -> XaiDeviceAuthorization:
    device_code = payload.get("device_code")
    user_code = payload.get("user_code")
    expires_in = payload.get("expires_in")
    if not isinstance(device_code, str) or not device_code or not isinstance(user_code, str) or not user_code:
        raise AuthError("xAI device authorization response is missing required fields.", provider="xai")
    if not isinstance(expires_in, (int, float)) or isinstance(expires_in, bool) or expires_in <= 0:
        raise AuthError("xAI device authorization response is missing expires_in.", provider="xai")
    interval = payload.get("interval")
    interval_s = (
        float(interval)
        if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0
        else 5.0
    )
    complete = payload.get("verification_uri_complete")
    return XaiDeviceAuthorization(
        user_code=user_code,
        verification_uri=_https_or_raise(payload.get("verification_uri")),
        verification_uri_complete=_https_or_raise(complete) if isinstance(complete, str) and complete else None,
        interval_s=interval_s,
        expires_in_s=float(expires_in),
        device_code=device_code,
    )


def _post_form_tolerant(url: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """POST a form; return (ok, parsed body) even for 4xx responses.

    Device-code polling encodes protocol state (authorization_pending,
    slow_down, ...) in 4xx bodies, so HTTP errors are data here.
    """
    try:
        return True, _post_form(url, payload)
    except urllib.error.HTTPError as exc:
        try:
            data = json.loads(exc.read().decode("utf-8"))
        except Exception:
            data = {}
        return False, data if isinstance(data, dict) else {}


def start_xai_device_login() -> XaiDeviceAuthorization:
    """Request a device authorization; show the user code, then poll."""
    ok, payload = _post_form_tolerant(
        XAI_DEVICE_CODE_URL,
        {"client_id": XAI_CLIENT_ID, "scope": XAI_OAUTH_SCOPE, "referrer": "lm15"},
    )
    if not ok:
        detail = payload.get("error_description") or payload.get("error") or "request failed"
        raise AuthError(f"xAI device authorization failed: {detail}", provider="xai")
    return _parse_xai_device_authorization(payload)


def poll_xai_device_login(
    device: XaiDeviceAuthorization,
    *,
    sleep: Any = time.sleep,
) -> LocalOAuthCredential:
    """Poll the token endpoint until the user approves; return the credential."""
    from .authkit import DeviceComplete, DeviceFailed, DevicePending, DeviceSlowDown, poll_device_code

    def _poll() -> Any:
        ok, payload = _post_form_tolerant(
            XAI_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_CLIENT_ID,
                "device_code": device.device_code,
            },
        )
        if ok:
            return DeviceComplete(_xai_credential_from_token_response(payload))
        error = payload.get("error")
        if error == "authorization_pending":
            return DevicePending()
        if error == "slow_down":
            interval = payload.get("interval")
            return DeviceSlowDown(
                float(interval)
                if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0
                else None
            )
        if error in {"access_denied", "authorization_denied"}:
            return DeviceFailed("xAI device authorization was denied.")
        if error == "expired_token":
            return DeviceFailed("xAI device code expired before it was approved.")
        detail = payload.get("error_description") or error or "request failed"
        return DeviceFailed(f"xAI device token polling failed: {detail}")

    return poll_device_code(
        _poll,
        interval_s=device.interval_s,
        expires_in_s=device.expires_in_s,
        provider="xai",
        wait_before_first_poll=True,
        sleep=sleep,
    )


def login_xai(
    auth_path: str | os.PathLike[str] | None = None,
    *,
    echo: Any = print,
) -> LocalOAuthCredential:
    """Interactive device-code login; persists the credential and returns it.

    Prints the verification URL and user code via ``echo`` and blocks until
    the user approves in a browser (or the code expires).  The credential is
    written to lm15's own store unless ``auth_path`` overrides it.
    """
    device = start_xai_device_login()
    target = device.verification_uri_complete or device.verification_uri
    echo(f"Open {target} and enter code: {device.user_code}")
    credential = poll_xai_device_login(device)
    write_xai_credential(credential, auth_path)
    return credential


# ─── Uniform login entry point ───────────────────────────────────
#
# One door, provider-dispatched.  Only providers with an lm15-owned flow
# actually log in here (today: xai).  Every other provider fails typed,
# naming the real path: the foreign CLI command that owns the flow, or the
# console URL where keys are created.  These URLs are guidance strings, not
# wire facts — they can drift, and drifting costs a stale hint, not broken
# inference.

_KEY_CONSOLE_URLS: dict[str, str] = {
    "openai": "https://platform.openai.com/api-keys",
    "openai-chat": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com",
    "gemini": "https://aistudio.google.com/apikey",
    "groq": "https://console.groq.com/keys",
    "openrouter": "https://openrouter.ai/keys",
    "xai": "https://console.x.ai",
}

_CLI_LOGIN_HINTS: dict[str, str] = {
    "claude-code": CLAUDE_CODE_LOGIN_HINT,
    "openai-codex": OPENAI_CODEX_LOGIN_HINT,
}

_KEYLESS_LOCAL_SERVERS = frozenset({"ollama", "vllm", "sglang"})


def login(
    provider: str,
    *,
    credentials_path: str | os.PathLike[str] | None = None,
    echo: Any = print,
) -> LocalOAuthCredential:
    """Run the login flow lm15 owns for ``provider``; fail typed otherwise.

    This is the uniform door: ``login("xai")`` runs the device-code flow and
    returns the stored credential.  Providers whose login lives elsewhere
    raise :class:`lm15.errors.UnsupportedFeatureError` naming the exact fix
    — the foreign CLI command (Claude Code, Codex) or the console URL where
    an API key is created.  Nothing here prompts, opens a browser, or
    spends money except the one flow you explicitly asked for.
    """
    canonical = provider.replace("_", "-")
    if canonical == "xai":
        return login_xai(credentials_path, echo=echo)
    if canonical in _CLI_LOGIN_HINTS:
        raise UnsupportedFeatureError(
            f"lm15 does not own the {canonical!r} login flow — the provider CLI does. "
            f"{_CLI_LOGIN_HINTS[canonical]}",
            provider=canonical,
        )
    if canonical in _KEYLESS_LOCAL_SERVERS:
        raise UnsupportedFeatureError(
            f"{canonical!r} is a keyless local server — there is nothing to log into. "
            "The router sends the placeholder key the server expects.",
            provider=canonical,
        )
    if canonical in _KEY_CONSOLE_URLS:
        raise UnsupportedFeatureError(
            f"{canonical!r} offers no OAuth login flow — only manually created API keys. "
            f"Create one at {_KEY_CONSOLE_URLS[canonical]} and set it in the environment "
            f"or RouterConfig(api_keys={{{canonical!r}: \"...\"}}).",
            provider=canonical,
        )
    raise UnsupportedFeatureError(
        f"lm15 has no login flow for {provider!r}. Supply an API key via the "
        "environment or RouterConfig(api_keys=...).",
        provider=canonical,
    )
