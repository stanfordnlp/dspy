"""
lm15.login.engine — the machinery every provider flow runs on.

spec/auth.md AUTH-18 (connected login, device polling, browser/OAuth
protections), AUTH-20 (bounded exchanges, uncertainty), AUTH-21 (what a
diagnostic may carry).  Provider flows (``lm15.login.flows``) describe
*their* protocol; this module supplies the parts that must be identical
for all of them:

- :class:`LoginContext` — one attempt's deadline, cancellation and UI, plus
  the clock/sleep/HTTP seams tests inject;
- :func:`http_json` / :func:`http_form` — bounded (30 s, 1 MiB), TLS-only
  exchanges whose failures never carry a token or a reflected provider
  string into an exception;
- :func:`run_device_flow` — RFC 8628 polling with the ratified pacing
  (provider interval or 5 s; ``slow_down`` adds at least 5 s; never
  decreases; deadline never extended);
- :class:`CallbackListener` — a one-shot loopback listener for
  authorization-code returns: loopback bind only, exact path, state checked
  on success *and* error returns, bounded request size, no access log;
- :func:`race_callback_and_manual` — the listener raced against a manual
  paste prompt, the loser dismissed;
- :func:`parse_manual_return` — a pasted redirect URL or bare code, checked
  against the attempt's registered return context.

Nothing here knows a provider's URL, client id or token shape.
"""

from __future__ import annotations

import html
import http.client
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit

from .._version import __version__
from ..errors import AuthError, AuthOperationError, ServerError, TransportError
from .types import AuthUI, InfoNotice, ManualCodePrompt, Notice, Prompt

__all__ = [
    "ATTEMPT_LIFETIME_S",
    "AUTH_RESPONSE_LIMIT",
    "CallbackListener",
    "CallbackReturn",
    "DeviceStep",
    "EXCHANGE_TIMEOUT_S",
    "HttpReply",
    "LoginCancelled",
    "LoginContext",
    "LoginDenied",
    "LoginExpired",
    "http_form",
    "http_json",
    "parse_manual_return",
    "await_return",
    "race_callback_and_manual",
    "run_device_flow",
]

ATTEMPT_LIFETIME_S = 15 * 60        # AUTH-18, R9
EXCHANGE_TIMEOUT_S = 30.0           # AUTH-20.5, R9
DEVICE_DEFAULT_INTERVAL_S = 5.0     # RFC 8628 §3.2
DEVICE_SLOW_DOWN_STEP_S = 5.0       # RFC 8628 §3.5
AUTH_RESPONSE_LIMIT = 1024 * 1024   # AUTH-18: 1 MiB auth HTTP body
CALLBACK_TARGET_LIMIT = 8 * 1024    # AUTH-18: 8 KiB request target
CALLBACK_HEADER_LIMIT = 32 * 1024   # AUTH-18: 32 KiB callback headers


# ─── Outcomes the engine raises; the manager maps them (AUTH-24) ──────


class LoginCancelled(Exception):
    """The caller, the UI or the deadline stopped the attempt.  Not an
    LM15Error: the conformance outcome is ``cancelled`` (AUTH-24)."""


class LoginExpired(Exception):
    """The attempt's 15-minute (or provider) deadline passed."""


class LoginDenied(Exception):
    """A validated provider denial (``access_denied``, ``invalid_grant`` on
    a code exchange, a device code the provider says expired).  The message
    is ours; provider text is never copied into it."""

    def __init__(self, message: str, *, status: int | None = None,
                 provider_code: str | None = None, stage: str = "authorization") -> None:
        super().__init__(message)
        self.status = status
        self.provider_code = provider_code
        self.stage = stage


# ─── The attempt context ───────────────────────────────────────────────


@dataclass
class LoginContext:
    """Everything a flow may touch during one attempt.

    ``deadline`` is monotonic; ``cancel`` is the caller's event; ``ui`` is
    the application's adapter.  ``clock``/``sleep``/``opener`` are seams:
    tests inject them, production leaves the defaults.
    """

    ui: AuthUI
    deadline: float
    cancel: threading.Event | None = None
    provider: str = ""
    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    opener: Callable[[urllib.request.Request, float], Any] | None = None
    sleep: Callable[[float], None] | None = None  # test seam; production waits on the cancel event
    _prompts_open: list[Prompt] = field(default_factory=list, repr=False)

    def remaining(self) -> float:
        return self.deadline - self.clock()

    def check(self) -> None:
        """Raise if the attempt should stop.  Called before every external
        step and before every wait."""
        if self.cancel is not None and self.cancel.is_set():
            raise LoginCancelled("login cancelled")
        if self.remaining() <= 0:
            raise LoginExpired("login attempt deadline reached")

    def wait(self, seconds: float) -> None:
        """Sleep, but wake on cancel; never past the deadline."""
        self.check()
        seconds = min(max(seconds, 0.0), max(self.remaining(), 0.0))
        if seconds <= 0:
            self.check()
            return
        if self.sleep is not None:
            self.sleep(seconds)
        elif self.cancel is not None:
            if self.cancel.wait(seconds):
                raise LoginCancelled("login cancelled")
        else:
            time.sleep(seconds)
        self.check()

    def notify(self, notice: Notice) -> None:
        self.ui.notify(notice)

    def prompt(self, prompt: Prompt) -> str:
        self.check()
        try:
            answer = self.ui.prompt(prompt)
        except (KeyboardInterrupt, EOFError) as exc:
            raise LoginCancelled("login cancelled at the prompt") from exc
        if not isinstance(answer, str):
            raise TypeError("AuthUI.prompt must return a str")
        return answer

    def budget(self) -> float:
        """The network budget for one exchange: 30 s, bounded by the deadline."""
        return max(0.1, min(EXCHANGE_TIMEOUT_S, self.remaining()))


# ─── Bounded HTTP (AUTH-18/20/21) ──────────────────────────────────────


# Only these fixed protocol words can leave a private auth response. Never
# reflect arbitrary error descriptions, URLs, header values or response bodies.
_OAUTH_ERROR_CODES = frozenset({
    "invalid_request", "invalid_client", "invalid_grant", "unauthorized_client",
    "unsupported_grant_type", "invalid_scope", "access_denied", "server_error",
    "temporarily_unavailable", "authorization_pending", "slow_down", "expired_token",
})


@dataclass(frozen=True, slots=True)
class HttpReply:
    status: int
    body: dict[str, Any] = field(repr=False)  # may hold tokens: never rendered
    ok: bool = False
    response_format: str = "unknown"
    oauth_error: str | None = None
    security_challenge: bool = False

    def failure_summary(self) -> str:
        details = [f"HTTP {self.status}", f"response={self.response_format}"]
        if self.oauth_error:
            details.append(f"OAuth error={self.oauth_error}")
        else:
            details.append("no recognized OAuth error code; cause not established")
        if self.security_challenge:
            details.append("response explicitly marked as a security challenge")
        elif self.response_format == "html":
            details.append("HTML alone does not establish a security block")
        return "; ".join(details)


def _auth_json_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate member in auth response")
        result[key] = value
    return result


def _reject_auth_number(value: str) -> None:
    raise ValueError("non-finite number in auth response")


def _tls_only(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise AuthOperationError(
            "refusing a credential-bearing exchange over a non-HTTPS URL",
            reason="method_unavailable", stage="exchange", recovery="operator_action",
        )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """AUTH-20.9: an unexpected redirect with credentials is refused."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


def _send(ctx: LoginContext, request: urllib.request.Request) -> HttpReply:
    _tls_only(request.full_url)
    # Identify this SDK, not urllib (some auth endpoints reject that default).
    # Keep an explicit provider-required identification, e.g. Copilot's headers.
    if not request.has_header("User-agent"):
        request.add_header("User-Agent", f"lm15/{__version__}")
    timeout = ctx.budget()
    opener = ctx.opener
    response_headers: Any = {}
    try:
        if opener is not None:
            response = opener(request, timeout)
        else:
            response = _OPENER.open(request, timeout=timeout)  # noqa: S310 - TLS enforced above
        with response:
            status = getattr(response, "status", None) or response.getcode()
            response_headers = getattr(response, "headers", None) or {}
            raw = response.read(AUTH_RESPONSE_LIMIT + 1)
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = exc.headers or {}
        try:
            raw = exc.read(AUTH_RESPONSE_LIMIT + 1)
        except Exception:
            raw = b""
        finally:
            exc.close()
    except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
        # The URL is safe; the exception text can contain anything the
        # network stack saw, so only its class is named.  ``exchange_uncertain``
        # is AUTH-20.6's distinction: a refused connection or a DNS failure
        # never reached the provider (safe to retry deliberately); a timeout
        # or a dropped connection after sending may have (indeterminate).
        error = TransportError(
            f"{ctx.provider or 'auth'}: network failure during an authentication exchange "
            f"({type(exc).__name__}) to {request.full_url}",
            provider=ctx.provider or None,
        )
        error.exchange_uncertain = _uncertain(exc)
        raise error from None
    if len(raw) > AUTH_RESPONSE_LIMIT:
        raise AuthError(
            f"{ctx.provider or 'auth'}: authentication response exceeded {AUTH_RESPONSE_LIMIT} bytes; refused",
            provider=ctx.provider or None,
        )
    body: dict[str, Any] = {}
    # Map headers to fixed categories; never return their raw values.
    content_type = response_headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    response_format = "empty"
    oauth_error = None
    if raw:
        response_format = "text_or_binary"
        if content_type in ("text/html", "application/xhtml+xml"):
            response_format = "html"
        elif content_type == "application/json" or content_type.endswith("+json"):
            response_format = "invalid_json"
        try:
            parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_auth_json_members,
                                parse_constant=_reject_auth_number)
        except (ValueError, UnicodeDecodeError, RecursionError):
            parsed = None
        else:
            response_format = "json"
        if isinstance(parsed, dict):
            body = parsed
            candidate = parsed.get("error")
            if isinstance(candidate, dict):
                candidate = candidate.get("code", candidate.get("type"))
            if isinstance(candidate, str) and candidate in _OAUTH_ERROR_CODES:
                oauth_error = candidate
    security_challenge = response_headers.get("cf-mitigated", "").strip().lower() == "challenge"
    if status >= 500:
        raise ServerError(
            f"{ctx.provider or 'auth'}: the authentication server answered HTTP {status}",
            provider=ctx.provider or None, status=status,
        )
    return HttpReply(status=status, body=body, ok=200 <= status < 300,
                     response_format=response_format, oauth_error=oauth_error,
                     security_challenge=security_challenge)


def _uncertain(exc: BaseException) -> bool:
    import socket

    reason = getattr(exc, "reason", exc)
    if isinstance(reason, (socket.gaierror, ConnectionRefusedError)):
        return False
    if isinstance(reason, (TimeoutError, socket.timeout, http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError)):
        return True
    return True  # unknown: conservative


def http_json(ctx: LoginContext, url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None,
              method: str = "POST") -> HttpReply:
    data = json.dumps(payload).encode("utf-8") if method != "GET" else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json", **(headers or {})},
    )
    return _send(ctx, request)


def http_form(ctx: LoginContext, url: str, payload: dict[str, str], *, headers: dict[str, str] | None = None) -> HttpReply:
    request = urllib.request.Request(
        url, data=urllib.parse.urlencode(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json", **(headers or {})},
    )
    return _send(ctx, request)


def http_get(ctx: LoginContext, url: str, *, headers: dict[str, str] | None = None) -> HttpReply:
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json", **(headers or {})})
    return _send(ctx, request)


# ─── Device flow (RFC 8628, AUTH-18) ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class DeviceStep:
    """What one poll observed.  ``complete`` carries the flow's value."""

    status: str  # pending | slow_down | complete | denied | expired
    value: Any = field(default=None, repr=False)
    interval_s: float | None = None


def run_device_flow(
    ctx: LoginContext,
    poll: Callable[[], DeviceStep],
    *,
    interval_s: float | None,
    expires_in_s: float | None,
    wait_before_first_poll: bool = True,
) -> Any:
    """Poll until complete.  Pacing per AUTH-18: the provider's interval or
    5 s; ``slow_down`` never shortens the interval and adds at least 5 s;
    the provider's expiry bounds the attempt but never extends it."""
    interval = float(interval_s) if interval_s and interval_s > 0 else DEVICE_DEFAULT_INTERVAL_S
    interval = max(interval, 1.0)
    if expires_in_s is not None and expires_in_s > 0:
        ctx.deadline = min(ctx.deadline, ctx.clock() + float(expires_in_s))
    if wait_before_first_poll:
        ctx.wait(interval)
    while True:
        ctx.check()
        step = poll()
        if step.status == "complete":
            return step.value
        if step.status == "denied":
            raise LoginDenied("the provider reported that authorization was denied")
        if step.status == "expired":
            raise LoginExpired("the provider reported that the device code expired")
        if step.status == "slow_down":
            proposed = step.interval_s if step.interval_s and step.interval_s > 0 else 0.0
            interval = max(interval + DEVICE_SLOW_DOWN_STEP_S, proposed)
        elif step.status != "pending":
            raise TypeError(f"device poll returned unknown status {step.status!r}")
        ctx.wait(interval)


# ─── Loopback callback listener (AUTH-18 browser protections) ─────────


@dataclass(frozen=True, slots=True)
class CallbackReturn:
    code: str = field(repr=False)
    state: str | None = None


_PAGE = (
    "<!doctype html><meta charset='utf-8'><meta name='referrer' content='no-referrer'>"
    "<title>{title}</title><p>{message}</p>"
)


class CallbackListener:
    """One-shot loopback listener for an authorization-code return.

    ``bind_host`` is the address the socket binds (``127.0.0.1`` or ``::1``;
    nothing else is accepted).  ``redirect_uri`` is the *registered* return
    URI the provider was told (it may say ``localhost``): the two are
    deliberately separate values (AUTH-18).  ``port=0`` picks an ephemeral
    port; a registered fixed port that is busy raises ``method_unavailable``
    so the caller can offer manual return instead.
    """

    def __init__(self, *, path: str, expected_state: str | None, bind_host: str = "127.0.0.1",
                 port: int = 0, redirect_host: str | None = None) -> None:
        if bind_host not in ("127.0.0.1", "::1"):
            raise AuthOperationError(
                f"callback listener may bind loopback only, not {bind_host!r}",
                reason="method_unavailable", stage="reservation", recovery="operator_action",
            )
        if not path.startswith("/"):
            raise ValueError("callback path must start with '/'")
        self._path = path
        self._expected_state = expected_state
        self._result: CallbackReturn | None = None
        self._denied = False
        self._done = threading.Event()
        self._lock = threading.Lock()
        listener = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "lm15"
            sys_version = ""

            def log_message(self, *_args: Any) -> None:  # AUTH-21: no access log, URLs carry codes
                pass

            def do_GET(self) -> None:  # noqa: N802
                if len(self.requestline) > CALLBACK_TARGET_LIMIT or sum(
                    len(k) + len(v) for k, v in self.headers.items()
                ) > CALLBACK_HEADER_LIMIT:
                    self._page(414, "Rejected", "Request too large.")
                    return
                split = urlsplit(self.path)
                if split.path != listener._path:
                    self._page(404, "Not found", "Callback route not found.")
                    return
                with listener._lock:
                    if listener._done.is_set():
                        self._page(409, "Already used", "This sign-in return was already handled.")
                        return
                    params = parse_qsl(split.query, keep_blank_values=True)
                    names = [k for k, _ in params]
                    if len(names) != len(set(names)):
                        self._page(400, "Rejected", "Sign-in return was not accepted.")
                        return
                    query = dict(params)
                    state = query.get("state")
                    if listener._expected_state is not None and state != listener._expected_state:
                        # Wrong state on a success OR an error return: generic
                        # rejection, and the legitimate wait continues.
                        self._page(400, "Rejected", "Sign-in return was not accepted.")
                        return
                    has_code = bool(query.get("code"))
                    has_error = "error" in query
                    if has_code == has_error:
                        # both, or neither: invalid
                        self._page(400, "Rejected", "Sign-in return was not accepted.")
                        return
                    if has_error:
                        listener._denied = True
                        listener._done.set()
                        self._page(400, "Not completed", "Sign-in was not completed.")
                        return
                    listener._result = CallbackReturn(code=query["code"], state=state)
                    listener._done.set()
                self._page(200, "Signed in", "Sign-in completed. You can close this window.")

            def _page(self, status: int, title: str, message: str) -> None:
                body = _PAGE.format(title=html.escape(title), message=html.escape(message)).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)

        family_server = HTTPServer
        if bind_host == "::1":
            import socket

            class _V6(HTTPServer):
                address_family = socket.AF_INET6

            family_server = _V6
        try:
            self._server = family_server((bind_host, port), _Handler)
        except OSError as exc:
            raise AuthOperationError(
                f"could not listen on {bind_host}:{port or 'ephemeral'} for the sign-in return "
                f"({exc.strerror or type(exc).__name__}); another program may be using the port",
                reason="method_unavailable", stage="reservation", recovery="choose_method",
            ) from None
        self._server.timeout = 0.25
        self._bind_host = bind_host
        self._redirect_host = redirect_host or bind_host
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def redirect_uri(self) -> str:
        host = self._redirect_host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}{self._path}"

    def start(self) -> None:
        def serve() -> None:
            while not self._done.is_set() and not self._closed:
                try:
                    self._server.handle_request()
                except Exception:
                    if self._closed:
                        return
        self._closed = False
        self._thread = threading.Thread(target=serve, name="lm15-login-callback", daemon=True)
        self._thread.start()

    def wait(self, ctx: LoginContext) -> CallbackReturn | None:
        """Block until a return arrives, the attempt is cancelled or the
        deadline passes.  ``None`` means the listener was stopped by the
        caller (the manual prompt won)."""
        while not self._done.is_set():
            if self._closed:
                return None
            ctx.check()
            self._done.wait(0.25)
        if self._denied:
            raise LoginDenied("the provider returned an error to the sign-in callback")
        return self._result

    def stop(self) -> None:
        self._closed = True
        self._done.set()
        try:
            self._server.server_close()
        except Exception:
            pass

    _closed = True

    def __enter__(self) -> "CallbackListener":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


# ─── Manual return parsing (AUTH-18) ───────────────────────────────────


def parse_manual_return(text: str, *, expected_state: str | None, allow_bare_code: bool,
                        registered_path: str | None, registered_uri: str | None = None) -> CallbackReturn:
    """Validate a manual return, including state on provider error callbacks.

    When registered_uri is supplied, a full URL must match its scheme, host,
    effective port and path. A bare code is allowed only for profiles that
    explicitly permit it; it is not permission for a URL to omit its state.
    No pasted value is included in an exception.
    """
    value = (text or "").strip()
    if not value:
        raise AuthOperationError("nothing was pasted", reason="invalid_login_state", stage="interaction",
                                 recovery="provide_input")
    if len(value) > CALLBACK_TARGET_LIMIT:
        raise AuthOperationError("pasted return is too long", reason="invalid_login_state", stage="interaction",
                                 recovery="provide_input")
    def invalid(message: str) -> AuthOperationError:
        return AuthOperationError(message, reason="invalid_login_state", stage="interaction",
                                  recovery="provide_input")

    code: str | None = None
    state: str | None = None
    params: list[tuple[str, str]] | None = None
    bare = False
    if "://" in value:
        try:
            split = urlsplit(value)
            if split.username is not None or split.password is not None or split.fragment:
                raise ValueError("unexpected URL components")
            if registered_path is not None and split.path != registered_path:
                raise ValueError("wrong callback path")
            if registered_uri is not None:
                expected = urlsplit(registered_uri)

                def address(url):
                    port = url.port if url.port is not None else {"https": 443, "http": 80}.get(url.scheme)
                    return url.scheme, url.hostname, port, url.path

                if address(split) != address(expected):
                    raise ValueError("wrong callback destination")
            params = parse_qsl(split.query, keep_blank_values=True)
        except ValueError:
            raise invalid("the pasted URL is not this sign-in's registered return URL") from None
    elif value.startswith(("code=", "state=", "error=")):
        params = parse_qsl(value, keep_blank_values=True)
    elif "#" in value:
        code, _, state = value.partition("#")
    else:
        code, bare = value, True

    denied = False
    if params is not None:
        names = [key for key, _ in params]
        if len(names) != len(set(names)):
            raise invalid("the pasted return repeats a parameter")
        query = dict(params)
        if "code" in query and "error" in query:
            raise invalid("the pasted return contains both a code and an error")
        denied = "error" in query
        code, state = query.get("code"), query.get("state")

    if bare and not allow_bare_code:
        raise invalid("paste the complete code#state or return URL, not the code alone")
    if expected_state is not None:
        if state is None:
            if not (bare and allow_bare_code):
                raise invalid("this provider's return must carry its state value")
        elif not secrets.compare_digest(state.encode("utf-8"), expected_state.encode("utf-8")):
            raise invalid("the pasted return does not belong to this sign-in attempt")
    # A wrong-state error must never terminate the legitimate attempt as denied.
    if denied:
        raise LoginDenied("the validated pasted return carries a provider error")
    if not code:
        raise invalid("no authorization code in the pasted text")
    return CallbackReturn(code=code, state=state)


# ─── Listener vs manual prompt (AUTH-16) ───────────────────────────────


def race_callback_and_manual(
    ctx: LoginContext,
    listener: CallbackListener | None,
    prompt: ManualCodePrompt,
    *,
    keep_listener: bool = False,
) -> tuple[CallbackReturn | None, str | None]:
    """Wait for whichever arrives first: the loopback return or a pasted
    value.  Returns ``(callback, None)`` or ``(None, pasted_text)``.  The
    loser is dismissed; a stale manual answer after a callback win is
    ignored (AUTH-16)."""
    if listener is None:
        return None, ctx.prompt(prompt)

    manual_box: dict[str, Any] = {}
    manual_done = threading.Event()

    def ask() -> None:
        try:
            manual_box["value"] = ctx.ui.prompt(prompt)
        except (KeyboardInterrupt, EOFError):
            manual_box["cancelled"] = True
        except BaseException as exc:  # a UI bug is a sanitized failure, not a crash into headless mode
            manual_box["error"] = type(exc).__name__
        finally:
            manual_done.set()

    thread = threading.Thread(target=ask, name="lm15-login-manual", daemon=True)
    thread.start()
    try:
        while True:
            ctx.check()
            if listener._done.is_set():
                result = listener.wait(ctx)
                dismiss = getattr(ctx.ui, "dismiss", None)
                if callable(dismiss):
                    dismiss(prompt)
                return result, None
            if manual_done.is_set():
                if not keep_listener or "value" not in manual_box:
                    listener.stop()
                if manual_box.get("cancelled"):
                    raise LoginCancelled("login cancelled at the prompt")
                if "error" in manual_box:
                    raise AuthOperationError(
                        f"the application's UI failed while prompting ({manual_box['error']})",
                        reason="interaction_required", stage="interaction", recovery="operator_action",
                    )
                return None, manual_box.get("value")
            manual_done.wait(0.25)
    finally:
        if not (keep_listener and "value" in manual_box and not listener._done.is_set()):
            listener.stop()


def await_return(
    ctx: LoginContext,
    listener: CallbackListener | None,
    prompt: ManualCodePrompt,
    parse: Callable[[str], CallbackReturn],
) -> CallbackReturn:
    """The authorization return, from the listener or a paste, validated.

    A paste that fails validation (wrong state, wrong URL, a bare code the
    profile does not accept) is rejected with a notice and the legitimate
    wait goes on — the listener keeps listening and the person is asked
    again (AUTH-18: a wrong return "does not terminate the legitimate
    wait"; AUTH-24 ``invalid_login_state``). Cancellation and the deadline
    still end it.
    """
    while True:
        returned, pasted = race_callback_and_manual(ctx, listener, prompt, keep_listener=True)
        if returned is not None:
            return returned
        try:
            return parse(pasted or "")
        except AuthOperationError as exc:
            if exc.reason != "invalid_login_state":
                raise
            ctx.notify(InfoNotice(f"{exc}. Try again."))
