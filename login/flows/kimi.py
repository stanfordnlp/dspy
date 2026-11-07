"""
lm15.login.flows.kimi — Kimi Code (subscription) device login.

RFC 8628 against ``auth.kimi.com`` (JSON replies).  The access token
authenticates ``https://api.kimi.com/coding`` as a bearer — a different
host from the metered ``moonshotai`` route, so this is its own route,
``kimi-code``, bound to that base URL.  Pi 0.87.0 ``oauth/kimi-coding.ts``
is the implementation reference.

**Availability: unverified** (no live LM15 receipt; the ``kimi-code`` route
is registered but has no wire receipt either — AUTH-26 says so).
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from ...credentials import BearerToken
from ..engine import DeviceStep, LoginContext, LoginDenied, http_form, run_device_flow
from ..types import DeviceCodeNotice, LoginMethod, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth, oauth_material

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
DEVICE_TIMEOUT_S = 15 * 60.0

METHOD_DEVICE = LoginMethod(
    id="device", label="Sign in with Kimi Code (subscription)", kind="account", flow="device_code",
    availability="unverified", reason="no live receipt yet", delivery=("device",), subscription=True,
)

DESCRIPTOR = ProviderDescriptor(
    id="kimi-code", label="Kimi Code (subscription)", service="Moonshot AI", routes=("kimi-code",),
    methods=(METHOD_DEVICE,),
)


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    return value if parsed.scheme in ("https", "http") and parsed.netloc else None


def _positive(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def _tokens(body: dict[str, Any], *, now_ms: int) -> Material:
    access, refresh, expires_in = body.get("access_token"), body.get("refresh_token"), body.get("expires_in")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise LoginDenied("Kimi Code token response is missing required fields")
    return oauth_material(access=access, refresh=refresh, expires_in_s=_positive(expires_in), now_ms=now_ms)


def _host(settings: dict[str, str]) -> str:
    return (settings.get("oauth_host") or DEFAULT_OAUTH_HOST).rstrip("/")


class KimiFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        host = _host(settings)
        reply = http_form(ctx, f"{host}/api/oauth/device_authorization", {"client_id": CLIENT_ID})
        if not reply.ok:
            raise LoginDenied(f"Kimi Code refused to start a device authorization (HTTP {reply.status})")
        body = reply.body
        device_code, user_code = body.get("device_code"), body.get("user_code")
        verification = _http_url(body.get("verification_uri_complete")) or _http_url(body.get("verification_uri"))
        if not isinstance(device_code, str) or not device_code or not isinstance(user_code, str) or not user_code or not verification:
            raise LoginDenied("Kimi Code device authorization response is missing required fields")
        interval, expires_in = _positive(body.get("interval")), _positive(body.get("expires_in")) or DEVICE_TIMEOUT_S
        ctx.notify(DeviceCodeNotice(user_code=user_code, verification_url=verification,
                                    expires_in_s=expires_in, interval_s=interval or 5.0))

        def poll() -> DeviceStep:
            reply = http_form(ctx, f"{host}/api/oauth/token", {
                "client_id": CLIENT_ID, "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            })
            if reply.ok and isinstance(reply.body.get("access_token"), str):
                return DeviceStep("complete", _tokens(reply.body, now_ms=int(ctx.wall_clock() * 1000)))
            error = reply.body.get("error")
            if error == "authorization_pending":
                return DeviceStep("pending")
            if error == "slow_down":
                return DeviceStep("slow_down", interval_s=_positive(reply.body.get("interval")))
            if error == "expired_token":
                return DeviceStep("expired")
            if error == "access_denied":
                return DeviceStep("denied")
            raise LoginDenied(f"Kimi Code device token request failed (HTTP {reply.status})")

        material = run_device_flow(ctx, poll, interval_s=interval, expires_in_s=expires_in)
        return LoginResult(material=material, label="Kimi Code subscription", renewal="refresh_token",
                           settings={"oauth_host": host} if host != DEFAULT_OAUTH_HOST else {})

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        refresh = material.get("refresh")
        if not isinstance(refresh, str) or not refresh:
            raise LoginDenied("Kimi Code credential has no refresh token")
        reply = http_form(ctx, f"{_host(settings)}/api/oauth/token", {
            "client_id": CLIENT_ID, "grant_type": "refresh_token", "refresh_token": refresh,
        })
        if reply.status in (401, 403) or reply.body.get("error") == "invalid_grant":
            raise LoginDenied(f"Kimi Code rejected the refresh token (HTTP {reply.status})")
        if not reply.ok:
            # 429/5xx are transient: the engine already raised ServerError for
            # 5xx; a 429 here is reported as a denial of *this* renewal, not
            # of the credential, by not marking needs_login (the manager only
            # marks on LoginDenied).
            from ...errors import RateLimitError

            raise RateLimitError(f"Kimi Code rate-limited the token refresh (HTTP {reply.status})",
                                 provider="kimi-code", status=reply.status)
        return LoginResult(material=_tokens(reply.body, now_ms=int(ctx.wall_clock() * 1000)),
                           label="Kimi Code subscription", renewal="refresh_token")

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        return RequestAuth(credential=BearerToken(material["access"]))
