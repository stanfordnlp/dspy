"""
lm15.authkit — building blocks for login flows. No UI, no opinions.

lm15 deliberately does not own a ``/login`` experience: apps and CLIs do.
This module gives those apps the pieces that are hard to get right and
painful to reimplement per language, all stdlib-only:

- :func:`generate_pkce` / :func:`pkce_challenge` — RFC 7636 (S256 only;
  ``plain`` is intentionally not offered because every current provider
  supports S256 and ``plain`` exists only as a downgrade).
- :func:`poll_device_code` — the RFC 8628 device-authorization polling state
  machine (pending / slow_down / complete / failure / expiry), with
  injectable clock and sleep so it is testable without real time.
- :class:`OAuthCallbackListener` — a one-shot loopback HTTP listener for
  authorization-code redirects, bound to ``127.0.0.1``, with state checking.
- :class:`CredentialFileStore` — an lm15-owned, locked, atomic, 0600
  credential file keyed by provider id (the ADC/pi ``auth.json`` pattern),
  with a serialized read-modify-write ``mutate`` as the only compound write
  path.

Secrets never appear in reprs or exception messages raised by this module.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Union
from urllib.parse import parse_qs, urlsplit

from ._authlock import hold_file_lock, write_private_json_atomic
from .errors import AuthError

__all__ = [
    "CredentialFileStore",
    "DeviceCodeExpiredError",
    "DeviceComplete",
    "DeviceFailed",
    "DevicePending",
    "DeviceSlowDown",
    "OAuthCallbackListener",
    "OAuthCallbackResult",
    "PKCEPair",
    "default_credentials_path",
    "generate_pkce",
    "pkce_challenge",
    "poll_device_code",
]


# ─── PKCE (RFC 7636, S256) ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PKCEPair:
    verifier: str = field(repr=False)  # secret until the code exchange
    challenge: str
    method: str = "S256"


def pkce_challenge(verifier: str) -> str:
    """S256 code challenge for ``verifier`` (RFC 7636 §4.2)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def generate_pkce() -> PKCEPair:
    """Fresh PKCE pair: 64 random bytes → 86-char base64url verifier."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    return PKCEPair(verifier=verifier, challenge=pkce_challenge(verifier))


# ─── Device-code polling (RFC 8628 §3.5) ─────────────────────────────


@dataclass(frozen=True, slots=True)
class DevicePending:
    """Authorization still pending; poll again after the interval."""


@dataclass(frozen=True, slots=True)
class DeviceSlowDown:
    """Server asked to slow down; interval grows by 5s unless it names one."""

    interval_s: float | None = None


@dataclass(frozen=True, slots=True)
class DeviceComplete:
    """Authorization finished; ``value`` is whatever the poll fn produced."""

    value: Any = field(repr=False)  # usually carries tokens


@dataclass(frozen=True, slots=True)
class DeviceFailed:
    """Terminal denial/failure. ``message`` must not contain secrets."""

    message: str


DevicePollResult = Union[DevicePending, DeviceSlowDown, DeviceComplete, DeviceFailed]

_SLOW_DOWN_STEP_S = 5.0  # RFC 8628 §3.5: "increase ... by 5 seconds"


class DeviceCodeExpiredError(AuthError):
    """The device code expired before the user approved it."""


def poll_device_code(
    poll: Callable[[], DevicePollResult],
    *,
    interval_s: float = 5.0,
    expires_in_s: float = 900.0,
    provider: str | None = None,
    wait_before_first_poll: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Run the device-authorization polling loop; return the completed value.

    ``poll`` performs one token-endpoint request and classifies the outcome.
    Raises :class:`DeviceCodeExpiredError` on expiry and :class:`AuthError`
    on terminal failure. ``sleep``/``clock`` are injectable for tests.
    """
    interval = max(interval_s, 0.0)
    deadline = clock() + expires_in_s
    first = True
    while True:
        if not first or wait_before_first_poll:
            if clock() + interval > deadline:
                raise DeviceCodeExpiredError(
                    "Device authorization expired before it was approved. Start the login again.",
                    provider=provider,
                )
            sleep(interval)
        first = False
        if clock() > deadline:
            raise DeviceCodeExpiredError(
                "Device authorization expired before it was approved. Start the login again.",
                provider=provider,
            )
        result = poll()
        if isinstance(result, DeviceComplete):
            return result.value
        if isinstance(result, DeviceFailed):
            raise AuthError(result.message, provider=provider)
        if isinstance(result, DeviceSlowDown):
            interval = result.interval_s if result.interval_s else interval + _SLOW_DOWN_STEP_S
        elif not isinstance(result, DevicePending):
            raise TypeError(f"poll() must return a DevicePollResult, got {type(result).__name__}")


# ─── Loopback callback listener ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OAuthCallbackResult:
    code: str = field(repr=False)  # authorization codes are secrets
    state: str | None = None


_SUCCESS_HTML = (
    "<!doctype html><meta charset='utf-8'><title>Signed in</title>"
    "<p>Authentication completed. You can close this window.</p>"
)
_ERROR_HTML = "<!doctype html><meta charset='utf-8'><title>Error</title><p>{message}</p>"


class OAuthCallbackListener:
    """One-shot loopback listener for an authorization-code redirect.

    Usage::

        with OAuthCallbackListener(expected_state=state) as listener:
            url = build_authorize_url(redirect_uri=listener.redirect_uri, ...)
            print(url)  # or open a browser
            result = listener.wait(timeout_s=300)

    Binds ``127.0.0.1`` (never a public interface) on an ephemeral port by
    default. Wrong-path and wrong-state requests get an error page and the
    listener keeps waiting; a provider ``error`` parameter or the deadline
    ends the wait. Single-threaded by design: build and display the
    authorization URL before calling :meth:`wait`.
    """

    def __init__(
        self,
        *,
        expected_state: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/callback",
    ) -> None:
        self._expected_state = expected_state
        self._path = path
        self._result: OAuthCallbackResult | None = None
        self._error: str | None = None
        listener = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # no request logging: URLs carry codes
                pass

            def do_GET(self) -> None:
                split = urlsplit(self.path)
                if split.path != listener._path:
                    self._reply(404, _ERROR_HTML.format(message="Callback route not found."))
                    return
                params = parse_qs(split.query)
                error = params.get("error", [None])[0]
                if error:
                    description = params.get("error_description", [error])[0]
                    listener._error = description or error
                    self._reply(400, _ERROR_HTML.format(message="Authorization was not completed."))
                    return
                state = params.get("state", [None])[0]
                if listener._expected_state is not None and state != listener._expected_state:
                    self._reply(400, _ERROR_HTML.format(message="State mismatch."))
                    return
                code = params.get("code", [None])[0]
                if not code:
                    self._reply(400, _ERROR_HTML.format(message="Missing authorization code."))
                    return
                listener._result = OAuthCallbackResult(code=code, state=state)
                self._reply(200, _SUCCESS_HTML)

            def _reply(self, status: int, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer((host, port), _Handler)

    @property
    def redirect_uri(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}{self._path}"

    def wait(self, *, timeout_s: float = 300.0) -> OAuthCallbackResult:
        """Block until the redirect arrives; raise on error or timeout."""
        deadline = time.monotonic() + timeout_s
        while self._result is None and self._error is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for the OAuth redirect.")
            self._server.timeout = min(remaining, 1.0)
            self._server.handle_request()
        if self._error is not None:
            raise AuthError(f"Authorization failed: {self._error}")
        assert self._result is not None
        return self._result

    def close(self) -> None:
        self._server.server_close()

    def __enter__(self) -> "OAuthCallbackListener":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ─── lm15-owned credential store ─────────────────────────────────────


def default_credentials_path() -> Path:
    """``$LM15_CREDENTIALS_PATH``, else ``$XDG_CONFIG_HOME/lm15/credentials.json``."""
    override = os.environ.get("LM15_CREDENTIALS_PATH")
    if override:
        return Path(override).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path("~/.config").expanduser()
    return base / "lm15" / "credentials.json"


class CredentialFileStore:
    """A locked, atomic, private (0600) credential file keyed by provider id.

    The file is one JSON object: ``{"<provider>": {...credential...}}``.
    ``mutate`` is the only compound write path — a serialized
    read-modify-write under the cross-process lock, so refresh flows can
    re-check state before writing (the double-checked pattern).

    The store deliberately does not interpret credential shapes; that is the
    caller's schema. It only guarantees storage semantics.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_credentials_path()

    def __repr__(self) -> str:  # never show file contents
        return f"CredentialFileStore(path={str(self.path)!r})"

    def _read_all(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except ValueError as exc:
            raise ValueError(f"Credential store at {self.path} is not valid JSON.") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Credential store at {self.path} must be a JSON object.")
        return data

    def read(self, provider: str) -> dict[str, Any] | None:
        """The stored credential for ``provider``, or None. Returns a copy."""
        credential = self._read_all().get(provider)
        return copy.deepcopy(credential) if isinstance(credential, dict) else None

    def list(self) -> tuple[str, ...]:
        """Provider ids with stored credentials. Never returns values."""
        return tuple(sorted(self._read_all()))

    def write(self, provider: str, credential: dict[str, Any]) -> None:
        self.mutate(provider, lambda _current: credential)

    def delete(self, provider: str) -> None:
        with hold_file_lock(self.path):
            data = self._read_all()
            if provider in data:
                del data[provider]
                write_private_json_atomic(self.path, data)

    def mutate(
        self,
        provider: str,
        fn: Callable[[dict[str, Any] | None], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        """Serialized read-modify-write for one provider's credential.

        ``fn`` receives the current credential (or None) as read inside the
        lock and returns the new credential, or None to leave the entry
        unchanged. Returns the post-write credential.
        """
        with hold_file_lock(self.path):
            data = self._read_all()
            current = data.get(provider)
            current = copy.deepcopy(current) if isinstance(current, dict) else None
            replacement = fn(current)
            if replacement is None:
                return current
            if not isinstance(replacement, dict):
                raise TypeError("mutate() callback must return a dict or None")
            data[provider] = replacement
            write_private_json_atomic(self.path, data)
            return copy.deepcopy(replacement)
