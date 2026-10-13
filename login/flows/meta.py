"""
lm15.login.flows.meta — Meta (Muse subscription) device login + key mint.

RFC 8628 device authorization against ``auth.meta.com`` (JSON replies);
the identity token it returns is **not** accepted for inference, so it is
exchanged at the Muse Code key-mint endpoint for a Model API key that
lives about a day.  Renewal is a re-mint with the stored identity token
(``renewal="remint"``); the identity token itself is not renewable, so a
401/403 from the mint is a permanent rejection (``needs_login``).  Pi
0.87.0 ``oauth/meta.ts`` is the implementation reference.

**Availability: unverified** (no live LM15 receipt).
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from ...credentials import ApiKey
from ..engine import DeviceStep, LoginContext, LoginDenied, http_form, http_json, run_device_flow
from ..types import DeviceCodeNotice, LoginMethod, ProgressNotice, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth

CLIENT_ID = "1031625952748946"  # Muse Code CLI
DEVICE_AUTHORIZATION_URL = "https://auth.meta.com/oidc/device/authorization/"
DEVICE_TOKEN_URL = "https://auth.meta.com/oidc/device/token/"
KEY_MINT_URL = "https://api.meta.ai/muse-code/key"
KEY_LIFETIME_S = 24 * 60 * 60.0

METHOD_DEVICE = LoginMethod(
    id="device", label="Sign in with Meta (Muse subscription)", kind="account", flow="device_code",
    availability="unverified", reason="no live receipt yet", delivery=("device",), subscription=True,
    billing_note="Minted Model API keys are tied to the Muse subscription; verify entitlement on your account.",
)

DESCRIPTOR = ProviderDescriptor(
    id="meta", label="Meta", service="Meta", routes=("meta", "meta-chat", "meta-anthropic"),
    methods=(
        METHOD_DEVICE,
        LoginMethod(id="api_key", label="API key from dev.meta.ai", kind="api_key", flow="form"),
    ),
    console_url="https://dev.meta.ai",
)


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    return value if parsed.scheme in ("https", "http") and parsed.netloc else None


def _positive(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def _mint(ctx: LoginContext, identity_token: str) -> Material:
    ctx.notify(ProgressNotice(stage="exchange", message="Enabling Meta Model API access…"))
    reply = http_json(ctx, KEY_MINT_URL, {}, headers={
        "Authorization": f"Bearer {identity_token}", "x-api-version": "1.0.0",
    })
    if reply.status in (401, 403):
        raise LoginDenied("Meta session is no longer valid; sign in again")
    if not reply.ok:
        raise LoginDenied(f"Meta API key mint failed (HTTP {reply.status})")
    key = reply.body.get("api_key")
    if not isinstance(key, str) or not key:
        action = _http_url(reply.body.get("action_url"))
        raise LoginDenied("Meta did not issue an API key" + (f"; complete setup at {action}" if action else ""))
    now_ms = int(ctx.wall_clock() * 1000)
    return {"type": "oauth", "access": key, "refresh": identity_token, "issued_at": now_ms,
            "lifetime_s": KEY_LIFETIME_S, "expires": int(now_ms + KEY_LIFETIME_S * 1000)}


class MetaFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        if method.id != "device":
            raise ValueError(method.id)
        reply = http_form(ctx, DEVICE_AUTHORIZATION_URL, {"client_id": CLIENT_ID})
        if not reply.ok:
            raise LoginDenied(f"Meta refused to start a device authorization (HTTP {reply.status})")
        body = reply.body
        device_code, user_code = body.get("device_code"), body.get("user_code")
        verification = _http_url(body.get("verification_uri_complete")) or _http_url(body.get("verification_uri"))
        if not isinstance(device_code, str) or not device_code or not isinstance(user_code, str) or not user_code or not verification:
            raise LoginDenied("Meta device authorization response is missing required fields")
        interval, expires_in = _positive(body.get("interval")), _positive(body.get("expires_in"))
        ctx.notify(DeviceCodeNotice(user_code=user_code, verification_url=verification,
                                    expires_in_s=expires_in or 900.0, interval_s=interval or 5.0))

        def poll() -> DeviceStep:
            reply = http_form(ctx, DEVICE_TOKEN_URL, {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code, "client_id": CLIENT_ID,
            })
            token = reply.body.get("access_token")
            if reply.ok and isinstance(token, str) and token:
                return DeviceStep("complete", token)
            error = reply.body.get("error")
            if error == "authorization_pending":
                return DeviceStep("pending")
            if error == "slow_down":
                return DeviceStep("slow_down", interval_s=_positive(reply.body.get("interval")))
            if error == "access_denied":
                return DeviceStep("denied")
            if error == "expired_token":
                return DeviceStep("expired")
            raise LoginDenied(f"Meta device token request failed (HTTP {reply.status})")

        identity = run_device_flow(ctx, poll, interval_s=interval, expires_in_s=expires_in)
        return LoginResult(material=_mint(ctx, identity), label="Meta (Muse subscription)", renewal="remint")

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        identity = material.get("refresh")
        if not isinstance(identity, str) or not identity:
            raise LoginDenied("Meta credential has no identity token to re-mint with")
        return LoginResult(material=_mint(ctx, identity), label="Meta (Muse subscription)", renewal="remint")

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        return RequestAuth(credential=ApiKey(material["access"]))
