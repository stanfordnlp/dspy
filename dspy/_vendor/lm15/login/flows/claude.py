"""
lm15.login.flows.claude — Claude subscription login owned by LM15.

``browser`` uses Claude's hosted return page and asks for the displayed
code#state (or the complete return URL). No local listener is needed: the
browser and Python may run on different machines. This matches the observed
Claude Code 2.1.280 manual login on 2026-09-23. ``loopback`` retains the
older local callback flow from the Pi 0.87.0 implementation reference.

**Availability: unverified** (AUTH-13.5, R1). Hosted LM15 login, inference,
fresh-process persistence and an early renewal were observed on 2026-09-23.
Provider permission and billing remain unresolved; loopback has no live LM15
receipt. Both methods still need explicit opt-in. Existing externally owned
CLI credentials remain available through ``lm15.login.flows.recipes``.
"""

from __future__ import annotations

import secrets
from typing import Any

from ...auth import CLAUDE_CODE_CLIENT_ID
from ...authkit import PKCEPair, generate_pkce, pkce_challenge
from ...credentials import BearerToken
from ...errors import AuthOperationError
from ..engine import (
    CallbackListener,
    LoginContext,
    LoginDenied,
    ManualCodePrompt,
    http_json,
    await_return,
    parse_manual_return,
)
from ..types import AuthUrlNotice, InfoNotice, LoginMethod, ProgressNotice, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth, oauth_material

CLIENT_ID = CLAUDE_CODE_CLIENT_ID
AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
LOOPBACK_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
LOOPBACK_REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"

METHOD_BROWSER = LoginMethod(
    id="browser", label="Sign in with Claude (paste code from hosted page)", kind="account", flow="authorization_code",
    availability="unverified",
    reason="LM15 hosted login, inference, persistence and early renewal observed 2026-09-23; permission and billing remain unverified",
    delivery=("manual",), subscription=True,
    billing_note="Provider permission and included usage must be verified separately for your account.",
)
METHOD_LOOPBACK = LoginMethod(
    id="loopback", label="Sign in with Claude (local browser callback)", kind="account", flow="authorization_code",
    availability="unverified", reason="no live LM15 receipt for this local callback flow",
    delivery=("loopback", "manual"), subscription=True,
    billing_note=METHOD_BROWSER.billing_note,
)

DESCRIPTOR = ProviderDescriptor(
    id="claude-code", label="Claude (subscription)", service="Anthropic", routes=("claude-code",),
    methods=(METHOD_BROWSER, METHOD_LOOPBACK,),
    docs_url="https://docs.claude.com",
)


def _tokens(body: dict[str, Any], *, now_ms: int) -> Material:
    access, refresh, expires_in = body.get("access_token"), body.get("refresh_token"), body.get("expires_in")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise LoginDenied("Claude token response is missing required fields")
    lifetime = float(expires_in) if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) and expires_in > 0 else None
    return oauth_material(access=access, refresh=refresh, expires_in_s=lifetime, now_ms=now_ms)


class ClaudeFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        if method.id not in ("browser", "loopback"):
            raise AuthOperationError("Unknown Claude login method", reason="method_unavailable",
                                     stage="discovery", recovery="choose_method", provider="claude-code")
        ctx.check()
        hosted = method.id == "browser"
        # 32 random bytes -> 43 characters, as in the captured native flow.
        # Independent state remains essential; it is not the PKCE verifier.
        verifier = secrets.token_urlsafe(32)
        pkce = PKCEPair(verifier=verifier, challenge=pkce_challenge(verifier)) if hosted else generate_pkce()
        state = secrets.token_urlsafe(32)
        redirect_uri = REDIRECT_URI if hosted else LOOPBACK_REDIRECT_URI
        authorize_url = AUTHORIZE_URL if hosted else LOOPBACK_AUTHORIZE_URL
        listener: CallbackListener | None = None
        try:
            if not hosted:
                try:
                    listener = CallbackListener(path=CALLBACK_PATH, expected_state=state, port=CALLBACK_PORT,
                                                redirect_host="localhost")
                    listener.start()
                except AuthOperationError as exc:
                    if exc.reason != "method_unavailable":
                        raise
                    ctx.notify(InfoNotice(f"Could not listen on port {CALLBACK_PORT}; "
                                          "paste the full redirect URL when the browser finishes."))
            query = {
                "code": "true", "client_id": CLIENT_ID, "response_type": "code", "redirect_uri": redirect_uri,
                "scope": SCOPES, "code_challenge": pkce.challenge, "code_challenge_method": "S256", "state": state,
            }
            import urllib.parse

            url = f"{authorize_url}?{urllib.parse.urlencode(query)}"
            instructions = (
                "Sign in to Claude in your browser. On the Authentication code page, copy the whole "
                "displayed code (including #state) and paste it here. The full return URL also works. "
                "Your browser may be on another machine; no localhost connection is needed."
                if hosted else
                "Sign in to Claude in your browser. If the local callback cannot be reached, "
                "paste the full redirect URL (or code#state) here."
            )
            ctx.notify(AuthUrlNotice(url=url, instructions=instructions))
            prompt = ManualCodePrompt(
                field_id="return", label="Paste the full code#state or return URL here",
                accepted="the full return URL, or code#state (a bare code without state is not accepted)",
            )
            returned = await_return(ctx, listener, prompt, lambda pasted: parse_manual_return(
                pasted, expected_state=state, allow_bare_code=False,
                registered_path=urllib.parse.urlsplit(redirect_uri).path, registered_uri=redirect_uri,
            ))
            ctx.check()
            ctx.notify(ProgressNotice(stage="exchange", message="Exchanging the authorization code…"))
            reply = http_json(ctx, TOKEN_URL, {
                "grant_type": "authorization_code", "code": returned.code, "redirect_uri": redirect_uri,
                "client_id": CLIENT_ID, "code_verifier": pkce.verifier, "state": state,
            })
            if not reply.ok:
                raise LoginDenied(
                    f"Claude authorization-code exchange failed: {reply.failure_summary()}. "
                    "The authorization code will not be retried automatically.",
                    status=reply.status, provider_code=reply.oauth_error, stage="exchange",
                )
            ctx.check()
            return LoginResult(material=_tokens(reply.body, now_ms=int(ctx.wall_clock() * 1000)),
                               label="Claude subscription", renewal="refresh_token")
        finally:
            if listener is not None:
                listener.stop()

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        refresh = material.get("refresh")
        if not isinstance(refresh, str) or not refresh:
            raise LoginDenied("Claude credential has no refresh token")
        reply = http_json(ctx, TOKEN_URL, {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh})
        if not reply.ok:
            raise LoginDenied(
                f"Claude token renewal failed: {reply.failure_summary()}",
                status=reply.status, provider_code=reply.oauth_error, stage="renewal",
            )
        return LoginResult(material=_tokens(reply.body, now_ms=int(ctx.wall_clock() * 1000)),
                           label="Claude subscription", renewal="refresh_token")

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        return RequestAuth(credential=BearerToken(material["access"]))
