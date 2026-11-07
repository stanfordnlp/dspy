"""
lm15.login.flows.meta — Meta (Muse Code subscription) device login + key mint.

RFC 8628 device authorization against ``auth.meta.com`` (JSON replies);
the identity token it returns is **not** accepted for inference, so it is
exchanged at the Muse Code key-mint endpoint for the account's Muse Code
Model API key.  Renewal is a re-mint with the stored identity token
(``renewal="remint"``); the identity token itself is not renewable, so a
401/403 from the mint is a permanent rejection (``needs_login``).  Muse
Code 1.4.6 is the reference (Pi 0.87.0 ``oauth/meta.ts`` was the first).

Live 2026-10-12 (lm15-contract ``receipts/2026-10-12-muse-code/``): the
mint answers the same key on every call, so a re-mint never revokes the
key Muse Code itself holds; the reply also names the API address
(``base_url``) and the subscription (``is_subs_active``,
``subs_tier_name``).  The key serves all three Meta wires, and
``muse-wonder-full-1.0`` (Power and Maximum Usage plans only) answers on
it.  The address is used as given when it is an https ``meta.ai`` host and
refused otherwise: the key is never sent anywhere else.

**Availability: unverified.**  Meta's Model API terms (2026-10-09, § 3.2)
say the Muse Code key "may not be used to access the Services outside the
coding harness"; lm15 is not that harness.  The descriptor says so.
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
KEY_LIFETIME_S = 24 * 60 * 60.0  # no expiry in the reply; a daily re-mint is free (same key, 2026-10-12)
TERMS_NOTE = ("Meta's Model API terms (§ 3.2) reserve the Muse Code key for Muse Code; using it here is outside "
              "those terms and puts the Meta account at risk. Subscription traffic is Contributor mode by default "
              "(Meta may train on it).")

METHOD_DEVICE = LoginMethod(
    id="device", label="Sign in with Meta (Muse subscription)", kind="account", flow="device_code",
    availability="unverified", reason="login, mint and inference observed 2026-10-12; " + TERMS_NOTE,
    delivery=("device",), subscription=True,
    billing_note="Billed to the Muse Code plan when it is active (muse-wonder-full-1.0 needs Power or Maximum "
                 "Usage); without one, Meta bills the key pay-as-you-go.",
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


def _api_base(value: Any) -> str | None:
    """The mint's ``base_url`` when the key may go there: https, a
    ``meta.ai`` host, no credentials in the URL.  ``None`` when absent."""
    if value is None:
        return None
    parsed = urllib.parse.urlsplit(value) if isinstance(value, str) else None
    host = (parsed.hostname or "") if parsed else ""
    if parsed is None or parsed.scheme != "https" or parsed.username or parsed.password or parsed.query \
            or not (host == "meta.ai" or host.endswith(".meta.ai")):
        raise LoginDenied("Meta named an API address outside meta.ai for the minted key; refusing to send it there")
    return value.rstrip("/")


def _plan(body: dict[str, Any]) -> str | None:
    """Display text for the account's plan; ``None`` when the reply is silent."""
    active = body.get("is_subs_active")
    if active is True:
        tier = body.get("subs_tier_name")
        return tier if isinstance(tier, str) and tier else "Muse Code subscription"
    if active is False:
        return "no Muse Code subscription (pay-as-you-go)"
    return None


def _result(material: Material, body: dict[str, Any]) -> LoginResult:
    return LoginResult(material=material, label="Meta (Muse subscription)", account_label=_plan(body),
                       renewal="remint")


def _mint(ctx: LoginContext, identity_token: str) -> tuple[Material, dict[str, Any]]:
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
    base_url = _api_base(reply.body.get("base_url"))
    now_ms = int(ctx.wall_clock() * 1000)
    material: Material = {"type": "oauth", "access": key, "refresh": identity_token, "issued_at": now_ms,
                          "lifetime_s": KEY_LIFETIME_S, "expires": int(now_ms + KEY_LIFETIME_S * 1000)}
    if base_url is not None:
        material["base_url"] = base_url
    return material, reply.body


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
        return _result(*_mint(ctx, identity))

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        identity = material.get("refresh")
        if not isinstance(identity, str) or not identity:
            raise LoginDenied("Meta credential has no identity token to re-mint with")
        return _result(*_mint(ctx, identity))

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        base_url = material.get("base_url")
        return RequestAuth(credential=ApiKey(material["access"]),
                           base_url=base_url if isinstance(base_url, str) and base_url else None)
