"""
lm15.login.flows.copilot — GitHub Copilot login.

GitHub's device flow (``login/device/code``) yields a GitHub token, which
is stored as ``refresh``; each request-time renewal exchanges it at
``copilot_internal/v2/token`` for a short-lived Copilot token whose
``proxy-ep`` field names the account's API host (AUTH-20.9: that host is
validated against GitHub's own domains before it is used as a base URL).
Enterprise accounts supply their GHE domain as a method field; it becomes
a provider *instance* setting, not a guess.

Deliberately **not** done here (AUTH-17, D17): Pi enables account model
policies during login (``POST /models/{id}/policy``).  That changes account
settings; LM15 does not do it inside authentication.  A future explicit
``enable_models`` operation may.

**Availability: unverified** pending provider-permission review. On
2026-09-23: github.com device login, Copilot token exchange, account model
catalog (59 models), gpt-4.1 inference, fresh-process persistence and one early
renewal were observed live. The ``github-copilot`` route remains a declared
provider (no contract wire receipt yet). GitHub requires a person to click
**Authorize** on the consent page; automation must not bypass that.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from ...credentials import BearerToken
from ..engine import DeviceStep, LoginContext, LoginDenied, http_form, http_get, run_device_flow
from ..types import DeviceCodeNotice, LoginMethod, MethodField, ProgressNotice, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth

CLIENT_ID = "Iv1.b507a08c87ecfe98"  # GitHub Copilot Chat's public OAuth app id
COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
DEFAULT_DOMAIN = "github.com"
DEFAULT_API_BASE = "https://api.individual.githubcopilot.com"

METHOD_DEVICE = LoginMethod(
    id="device", label="Sign in with GitHub (Copilot subscription)", kind="account", flow="device_code",
    availability="unverified",
    reason="Login, catalog, inference, persistence and early renewal observed 2026-09-23; permission review pending",
    delivery=("device",), subscription=True,
    fields=(MethodField(id="enterprise_domain", label="GitHub Enterprise domain (blank for github.com)",
                        required=False, help="e.g. company.ghe.com"),),
    billing_note="Some models require enabling on your account first; LM15 does not change that setting during login.",
)

DESCRIPTOR = ProviderDescriptor(
    id="github-copilot", label="GitHub Copilot", service="GitHub", routes=("github-copilot",),
    methods=(METHOD_DEVICE,),
)


def _domain(settings: dict[str, str]) -> str:
    raw = (settings.get("enterprise_domain") or "").strip()
    if not raw:
        return DEFAULT_DOMAIN
    parsed = urllib.parse.urlsplit(raw if "://" in raw else f"https://{raw}")
    host = parsed.hostname
    if not host or not re.fullmatch(r"[a-z0-9.-]+", host):
        raise LoginDenied("invalid GitHub Enterprise domain")
    return host


def _urls(domain: str) -> dict[str, str]:
    return {
        "device": f"https://{domain}/login/device/code",
        "token": f"https://{domain}/login/oauth/access_token",
        "copilot": f"https://api.{domain}/copilot_internal/v2/token",
    }


def base_url_for(material: Material, settings: dict[str, str]) -> str:
    """The account's API host from the Copilot token, validated against
    GitHub's domains; never an arbitrary host a token string names."""
    token = material.get("access", "")
    match = re.search(r"proxy-ep=([^;]+)", token if isinstance(token, str) else "")
    domain = _domain(settings)
    if match:
        host = match.group(1).strip().lower()
        api_host = re.sub(r"^proxy\.", "api.", host)
        allowed = (".githubcopilot.com",) if domain == DEFAULT_DOMAIN else (f".{domain}", ".githubcopilot.com")
        if re.fullmatch(r"[a-z0-9.-]+", api_host) and any(api_host.endswith(suffix) for suffix in allowed):
            return f"https://{api_host}"
    if domain != DEFAULT_DOMAIN:
        return f"https://copilot-api.{domain}"
    return DEFAULT_API_BASE


def _exchange(ctx: LoginContext, github_token: str, settings: dict[str, str]) -> Material:
    reply = http_get(ctx, _urls(_domain(settings))["copilot"], headers={
        "Authorization": f"Bearer {github_token}", **COPILOT_HEADERS,
    })
    if reply.status in (401, 403):
        raise LoginDenied("GitHub rejected the token for Copilot; sign in again")
    if not reply.ok:
        raise LoginDenied(f"Copilot token exchange failed (HTTP {reply.status})")
    token, expires_at = reply.body.get("token"), reply.body.get("expires_at")
    if not isinstance(token, str) or not token or not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        raise LoginDenied("Copilot token response is missing required fields")
    now_ms = int(ctx.wall_clock() * 1000)
    expires_ms = int(expires_at * 1000)
    return {"type": "oauth", "access": token, "refresh": github_token, "issued_at": now_ms,
            "lifetime_s": max((expires_ms - now_ms) / 1000.0, 1.0), "expires": expires_ms}


class CopilotFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        merged = dict(settings)
        if answers.get("enterprise_domain"):
            merged["enterprise_domain"] = answers["enterprise_domain"]
        domain = _domain(merged)
        urls = _urls(domain)
        reply = http_form(ctx, urls["device"], {"client_id": CLIENT_ID, "scope": "read:user"},
                          headers={"User-Agent": COPILOT_HEADERS["User-Agent"]})
        if not reply.ok:
            raise LoginDenied(f"GitHub refused to start a device authorization (HTTP {reply.status})")
        body = reply.body
        device_code, user_code, verification = body.get("device_code"), body.get("user_code"), body.get("verification_uri")
        if not isinstance(device_code, str) or not isinstance(user_code, str) or not isinstance(verification, str):
            raise LoginDenied("GitHub device authorization response is missing required fields")
        parsed = urllib.parse.urlsplit(verification)
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise LoginDenied("GitHub returned an untrusted verification URL")
        interval = body.get("interval")
        expires_in = body.get("expires_in")
        interval_s = float(interval) if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0 else None
        expires_s = float(expires_in) if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) and expires_in > 0 else None
        ctx.notify(DeviceCodeNotice(user_code=user_code, verification_url=verification,
                                    expires_in_s=expires_s or 900.0, interval_s=interval_s or 5.0))

        def poll() -> DeviceStep:
            reply = http_form(ctx, urls["token"], {
                "client_id": CLIENT_ID, "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }, headers={"User-Agent": COPILOT_HEADERS["User-Agent"]})
            token = reply.body.get("access_token")
            if isinstance(token, str) and token:
                return DeviceStep("complete", token)
            error = reply.body.get("error")
            if error == "authorization_pending":
                return DeviceStep("pending")
            if error == "slow_down":
                value = reply.body.get("interval")
                return DeviceStep("slow_down", interval_s=float(value) if isinstance(value, (int, float)) and value > 0 else None)
            if error == "expired_token":
                return DeviceStep("expired")
            if error == "access_denied":
                return DeviceStep("denied")
            raise LoginDenied(f"GitHub device authorization failed (HTTP {reply.status})")

        github_token = run_device_flow(ctx, poll, interval_s=interval_s, expires_in_s=expires_s)
        ctx.notify(ProgressNotice(stage="exchange", message="Exchanging the GitHub token for a Copilot token…"))
        material = _exchange(ctx, github_token, merged)
        label = "GitHub Copilot" if domain == DEFAULT_DOMAIN else f"GitHub Copilot ({domain})"
        return LoginResult(material=material, label=label, renewal="remint",
                           settings={"enterprise_domain": domain} if domain != DEFAULT_DOMAIN else {})

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        github_token = material.get("refresh")
        if not isinstance(github_token, str) or not github_token:
            raise LoginDenied("Copilot credential has no GitHub token to renew with")
        return LoginResult(material=_exchange(ctx, github_token, settings), label="GitHub Copilot", renewal="remint")

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        return RequestAuth(credential=BearerToken(material["access"]), headers=dict(COPILOT_HEADERS),
                           base_url=base_url_for(material, settings))
