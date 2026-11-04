"""
lm15.login.flows.xai — xAI subscription login (RFC 8628 device code).

Migrated from ``lm15.auth`` on 2026-09-22 (R2, R12): same protocol, same
client id, same store entry (``{"xai": {"type": "oauth", ...}}``), new
machinery underneath.  ``lm15.auth.login_xai`` still exists and now runs
this flow, so the legacy entry point and the managed manager share one
implementation.  Live receipt: the device flow, renewal and inference were
validated before this migration through ``lm15.auth`` (2026-09-01); the
managed path re-uses those exact requests and is marked ``supported`` on
that basis. The managed path itself was then observed live on 2026-09-22/23:
device login, inference, streaming, logout blocking the env key, fresh-process
persistence and one early refresh-token renewal (HTTP 200).
"""

from __future__ import annotations

from typing import Any

from ...credentials import BearerToken
from ..engine import DeviceStep, LoginContext, LoginDenied, http_form, run_device_flow
from ..types import DeviceCodeNotice, LoginMethod, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth, oauth_material

CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
TOKEN_URL = "https://auth.x.ai/oauth2/token"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
DEFAULT_LIFETIME_S = 3600.0  # xAI omits expires_in on some replies; the value Pi and lm15 assumed since 2026-09-01

METHOD_DEVICE = LoginMethod(
    id="device", label="Sign in with SuperGrok or X Premium", kind="account", flow="device_code",
    availability="supported", delivery=("device",), subscription=True,
    billing_note="Subscription access per xAI's own recommendation (2026-09-01); the API key path is metered.",
)

DESCRIPTOR = ProviderDescriptor(
    id="xai", label="xAI", service="xAI", routes=("xai",),
    methods=(
        METHOD_DEVICE,
        LoginMethod(id="api_key", label="API key from console.x.ai", kind="api_key", flow="form",
                    fields=(), billing_note="Metered per token."),
    ),
    console_url="https://console.x.ai",
)


def _https(value: Any) -> str:
    import urllib.parse

    if isinstance(value, str):
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme == "https" and parsed.netloc:
            return value
    raise LoginDenied("xAI returned an untrusted verification URL")


def _positive(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def _material_from_token(body: dict[str, Any], *, now_ms: int, previous_refresh: str | None) -> Material:
    access = body.get("access_token")
    if not isinstance(access, str) or not access:
        raise LoginDenied("xAI token response carried no access token")
    refresh = body.get("refresh_token")
    if not isinstance(refresh, str) or not refresh:
        refresh = previous_refresh  # xAI may omit it when it does not rotate
    lifetime = _positive(body.get("expires_in")) or DEFAULT_LIFETIME_S
    return oauth_material(access=access, refresh=refresh, expires_in_s=lifetime, now_ms=now_ms)


class XaiFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        if method.id != "device":
            raise ValueError(method.id)
        reply = http_form(ctx, DEVICE_CODE_URL, {"client_id": CLIENT_ID, "scope": SCOPE, "referrer": "lm15"})
        if not reply.ok:
            raise LoginDenied(f"xAI refused to start a device authorization (HTTP {reply.status})")
        body = reply.body
        device_code, user_code = body.get("device_code"), body.get("user_code")
        if not isinstance(device_code, str) or not device_code or not isinstance(user_code, str) or not user_code:
            raise LoginDenied("xAI device authorization response is missing required fields")
        verification = _https(body.get("verification_uri"))
        complete = body.get("verification_uri_complete")
        target = _https(complete) if isinstance(complete, str) and complete else verification
        interval, expires_in = _positive(body.get("interval")), _positive(body.get("expires_in"))
        ctx.notify(DeviceCodeNotice(user_code=user_code, verification_url=target,
                                    expires_in_s=expires_in or 900.0, interval_s=interval or 5.0))

        def poll() -> DeviceStep:
            reply = http_form(ctx, TOKEN_URL, {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": CLIENT_ID, "device_code": device_code,
            })
            if reply.ok:
                return DeviceStep("complete", _material_from_token(reply.body, now_ms=int(ctx.wall_clock() * 1000),
                                                                   previous_refresh=None))
            error = reply.body.get("error")
            if error == "authorization_pending":
                return DeviceStep("pending")
            if error == "slow_down":
                return DeviceStep("slow_down", interval_s=_positive(reply.body.get("interval")))
            if error in ("access_denied", "authorization_denied"):
                return DeviceStep("denied")
            if error == "expired_token":
                return DeviceStep("expired")
            raise LoginDenied(f"xAI device token polling failed (HTTP {reply.status})")

        material = run_device_flow(ctx, poll, interval_s=interval, expires_in_s=expires_in)
        return LoginResult(material=material, label="xAI subscription", renewal="refresh_token")

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        refresh = material.get("refresh")
        if not isinstance(refresh, str) or not refresh:
            raise LoginDenied("xAI credential has no refresh token")
        reply = http_form(ctx, TOKEN_URL, {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh})
        if not reply.ok:
            if reply.status in (400, 401, 403):
                raise LoginDenied(f"xAI rejected the refresh token (HTTP {reply.status})")
            raise LoginDenied(f"xAI refresh failed (HTTP {reply.status})")
        return LoginResult(
            material=_material_from_token(reply.body, now_ms=int(ctx.wall_clock() * 1000), previous_refresh=refresh),
            label="xAI subscription", renewal="refresh_token",
        )

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        return RequestAuth(credential=BearerToken(material["access"]))
