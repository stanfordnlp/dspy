"""
lm15.login.flows.codex — ChatGPT (Codex) subscription login owned by LM15.

Two methods: ``browser`` (authorization-code + PKCE, loopback return on the
registered ``localhost:1455`` redirect, manual paste alternative) and
``device`` (OpenAI's device-authorization endpoints, which return an
authorization code + verifier pair that is then exchanged like the browser
flow).  Pi 0.87.0 ``oauth/openai-codex.ts`` is the implementation
reference; the client id is the Codex CLI's.

**Availability: unverified** (AUTH-13.5, R1). On 2026-09-23, browser login,
model discovery, inference, fresh-process persistence and early renewal worked
with a managed grant. Device login and inference also worked with a separate
memory-only grant. Provider permission and billing remain unresolved; device
renewal/persistence were not separately exercised. Existing CLI access remains
available via the external source.
"""

from __future__ import annotations

import secrets
import urllib.parse
from typing import Any

from ...auth import extract_chatgpt_account_id, jwt_expires_at_ms
from ...authkit import generate_pkce
from ...credentials import BearerToken
from ..engine import (
    CallbackListener,
    DeviceStep,
    LoginContext,
    LoginDenied,
    ManualCodePrompt,
    http_form,
    http_json,
    await_return,
    parse_manual_return,
    run_device_flow,
)
from ..types import AuthUrlNotice, DeviceCodeNotice, InfoNotice, LoginMethod, ProgressNotice, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth, oauth_material

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE}/oauth/token"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
DEVICE_USER_CODE_URL = f"{AUTH_BASE}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URL = f"{AUTH_BASE}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE}/deviceauth/callback"
DEVICE_TIMEOUT_S = 15 * 60
SCOPE = "openid profile email offline_access"

_UNVERIFIED = "provider permission and billing remain unverified"

METHOD_BROWSER = LoginMethod(
    id="browser", label="Sign in with ChatGPT (browser)", kind="account", flow="authorization_code",
    availability="unverified", reason="Browser login, inference, persistence and early renewal observed 2026-09-23; " + _UNVERIFIED,
    delivery=("loopback", "manual"), subscription=True,
)
METHOD_DEVICE = LoginMethod(
    id="device", label="Sign in with ChatGPT (device code, for SSH/headless)", kind="account", flow="device_code",
    availability="unverified", reason="Device login and inference observed 2026-09-23; " + _UNVERIFIED,
    delivery=("device",), subscription=True,
)

DESCRIPTOR = ProviderDescriptor(
    id="openai-codex", label="ChatGPT (subscription)", service="OpenAI", routes=("openai-codex",),
    methods=(METHOD_BROWSER, METHOD_DEVICE),
)


def _tokens(body: dict[str, Any], *, now_ms: int) -> Material:
    access, refresh, expires_in = body.get("access_token"), body.get("refresh_token"), body.get("expires_in")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise LoginDenied("ChatGPT token response is missing required fields")
    lifetime = float(expires_in) if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) and expires_in > 0 else None
    if lifetime is None:
        exp = jwt_expires_at_ms(access)
        if exp is not None:
            lifetime = max((exp + 5 * 60 * 1000 - now_ms) / 1000.0, 0.0) or None
    account = extract_chatgpt_account_id(access)
    if not account:
        raise LoginDenied("ChatGPT token carries no account id")
    extra: dict[str, Any] = {"accountId": account}
    id_token = body.get("id_token")
    if isinstance(id_token, str) and id_token:
        extra["id_token"] = id_token
    return oauth_material(access=access, refresh=refresh, expires_in_s=lifetime, now_ms=now_ms, extra=extra)


class CodexFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        if method.id == "device":
            return self._login_device(ctx)
        if method.id == "browser":
            return self._login_browser(ctx)
        raise ValueError(method.id)

    def _exchange(self, ctx: LoginContext, code: str, verifier: str, redirect_uri: str) -> LoginResult:
        ctx.notify(ProgressNotice(stage="exchange", message="Exchanging the authorization code…"))
        reply = http_form(ctx, TOKEN_URL, {
            "grant_type": "authorization_code", "client_id": CLIENT_ID, "code": code,
            "code_verifier": verifier, "redirect_uri": redirect_uri,
        })
        if not reply.ok:
            raise LoginDenied(f"ChatGPT rejected the authorization code (HTTP {reply.status})")
        material = _tokens(reply.body, now_ms=int(ctx.wall_clock() * 1000))
        return LoginResult(material=material, label="ChatGPT subscription", renewal="refresh_token",
                           account_label=material.get("accountId"))

    def _login_browser(self, ctx: LoginContext) -> LoginResult:
        pkce = generate_pkce()
        state = secrets.token_hex(16)
        listener: CallbackListener | None
        try:
            listener = CallbackListener(path=CALLBACK_PATH, expected_state=state, port=CALLBACK_PORT,
                                        redirect_host="localhost")
            listener.start()
        except Exception as exc:
            listener = None
            ctx.notify(InfoNotice(f"Could not listen on port {CALLBACK_PORT} ({type(exc).__name__}); "
                                  "paste the redirect URL when the browser finishes."))
        try:
            query = {
                "response_type": "code", "client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI, "scope": SCOPE,
                "code_challenge": pkce.challenge, "code_challenge_method": "S256", "state": state,
                "id_token_add_organizations": "true", "codex_cli_simplified_flow": "true", "originator": "lm15",
            }
            url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(query)}"
            ctx.notify(AuthUrlNotice(url=url, instructions=(
                "Sign in to ChatGPT in your browser. If the browser is on another machine, paste the final "
                "redirect URL back here.")))
            prompt = ManualCodePrompt(field_id="return", label="Paste the redirect URL here (or wait for the browser)")
            returned = await_return(ctx, listener, prompt, lambda pasted: parse_manual_return(
                pasted, expected_state=state, allow_bare_code=False, registered_path=CALLBACK_PATH))
            return self._exchange(ctx, returned.code, pkce.verifier, REDIRECT_URI)
        finally:
            if listener is not None:
                listener.stop()

    def _login_device(self, ctx: LoginContext) -> LoginResult:
        reply = http_json(ctx, DEVICE_USER_CODE_URL, {"client_id": CLIENT_ID})
        if not reply.ok:
            if reply.status == 404:
                raise LoginDenied("ChatGPT device-code login is not enabled for this server; use the browser method")
            raise LoginDenied(f"ChatGPT refused to start a device authorization (HTTP {reply.status})")
        body = reply.body
        device_id, user_code = body.get("device_auth_id"), body.get("user_code")
        interval = body.get("interval")
        if isinstance(interval, str):
            try:
                interval = float(interval.strip())
            except ValueError:
                interval = None
        if not isinstance(device_id, str) or not device_id or not isinstance(user_code, str) or not user_code:
            raise LoginDenied("ChatGPT device authorization response is missing required fields")
        interval_s = float(interval) if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval >= 0 else None
        ctx.notify(DeviceCodeNotice(user_code=user_code, verification_url=DEVICE_VERIFICATION_URL,
                                    expires_in_s=DEVICE_TIMEOUT_S, interval_s=interval_s or 5.0))

        def poll() -> DeviceStep:
            reply = http_json(ctx, DEVICE_TOKEN_URL, {"device_auth_id": device_id, "user_code": user_code})
            if reply.ok:
                code, verifier = reply.body.get("authorization_code"), reply.body.get("code_verifier")
                if not isinstance(code, str) or not code or not isinstance(verifier, str) or not verifier:
                    raise LoginDenied("ChatGPT device token response is missing required fields")
                return DeviceStep("complete", (code, verifier))
            if reply.status in (403, 404):
                return DeviceStep("pending")
            error = reply.body.get("error")
            error_code = error.get("code") if isinstance(error, dict) else error
            if error_code == "deviceauth_authorization_pending":
                return DeviceStep("pending")
            if error_code == "slow_down":
                return DeviceStep("slow_down")
            raise LoginDenied(f"ChatGPT device authorization failed (HTTP {reply.status})")

        code, verifier = run_device_flow(ctx, poll, interval_s=interval_s, expires_in_s=DEVICE_TIMEOUT_S,
                                         wait_before_first_poll=True)
        return self._exchange(ctx, code, verifier, DEVICE_REDIRECT_URI)

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        refresh = material.get("refresh")
        if not isinstance(refresh, str) or not refresh:
            raise LoginDenied("ChatGPT credential has no refresh token")
        reply = http_form(ctx, TOKEN_URL, {"grant_type": "refresh_token", "refresh_token": refresh, "client_id": CLIENT_ID})
        if not reply.ok:
            raise LoginDenied(f"ChatGPT rejected the refresh token (HTTP {reply.status})")
        body = dict(reply.body)
        if not body.get("refresh_token"):
            body["refresh_token"] = refresh  # OpenAI may omit it when it does not rotate
        material = _tokens(body, now_ms=int(ctx.wall_clock() * 1000))
        return LoginResult(material=material, label="ChatGPT subscription", renewal="refresh_token",
                           account_label=material.get("accountId"))

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        account = material.get("accountId") or extract_chatgpt_account_id(material["access"])
        return RequestAuth(credential=BearerToken(material["access"]),
                           headers={"chatgpt-account-id": account} if account else {}, account_id=account)
