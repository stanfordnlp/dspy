"""
lm15.login.flows.openrouter — OpenRouter sign-in that mints an API key.

PKCE (S256) authorization at ``openrouter.ai/auth``; the return code is
exchanged at ``/api/v1/auth/keys`` for a **permanent, user-controlled API
key**, not an expiring token pair (AUTH-12: an account flow may produce a
key).  The callback is a one-shot loopback listener on an ephemeral port
with a random path, raced against a manual paste.  No client id: OpenRouter
binds the key to the callback URL and the PKCE verifier.

**Availability: unverified** pending broader conformance/support review.
Browser key issuance, limit inspection, model discovery, inference and
fresh-process persistence were observed on 2026-09-23 with a $1-limited key.
``kind=account`` means an authorization journey, not a subscription: requests
spend the user's OpenRouter credits (AUTH-13.7).
"""

from __future__ import annotations

import secrets
import urllib.parse
from typing import Any

from ...authkit import generate_pkce
from ...credentials import ApiKey
from ..engine import (
    CallbackListener,
    LoginContext,
    LoginDenied,
    ManualCodePrompt,
    http_json,
    await_return,
    parse_manual_return,
)
from ..types import AuthUrlNotice, LoginMethod, ProgressNotice, ProviderDescriptor
from .base import LoginResult, Material, ProviderFlow, RequestAuth

AUTHORIZE_URL = "https://openrouter.ai/auth"
KEY_URL = "https://openrouter.ai/api/v1/auth/keys"

METHOD_BROWSER = LoginMethod(
    id="browser", label="Sign in with OpenRouter (creates an API key for this app)", kind="account",
    flow="authorization_code", availability="unverified",
    reason="Login, key limit, inference and persistence observed 2026-09-23; broader support review pending",
    delivery=("loopback", "manual"), subscription=False,
    billing_note="The minted key spends your OpenRouter credits like any other key.",
)

DESCRIPTOR = ProviderDescriptor(
    id="openrouter", label="OpenRouter", service="OpenRouter", routes=("openrouter",),
    methods=(
        METHOD_BROWSER,
        LoginMethod(id="api_key", label="API key from openrouter.ai/keys", kind="api_key", flow="form"),
    ),
    console_url="https://openrouter.ai/keys",
)


def _exchange(ctx: LoginContext, code: str, verifier: str) -> Material:
    reply = http_json(ctx, KEY_URL, {"code": code, "code_verifier": verifier, "code_challenge_method": "S256"})
    if not reply.ok:
        raise LoginDenied(f"OpenRouter rejected the authorization code (HTTP {reply.status})")
    key = reply.body.get("key")
    if not isinstance(key, str) or not key:
        raise LoginDenied("OpenRouter returned no key")
    return {"type": "api_key", "key": key, "minted": True}


class OpenRouterFlow(ProviderFlow):
    descriptor = DESCRIPTOR

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        if method.id != "browser":
            raise ValueError(method.id)
        pkce = generate_pkce()
        path = f"/oauth/callback/{secrets.token_urlsafe(24)}"
        # No state parameter in OpenRouter's protocol: the one-time random
        # callback path plus PKCE is the evidenced equivalent binding (AUTH-18).
        listener = CallbackListener(path=path, expected_state=None, port=0)
        listener.start()
        try:
            query = {"callback_url": listener.redirect_uri, "code_challenge": pkce.challenge, "code_challenge_method": "S256"}
            url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(query)}"
            ctx.notify(AuthUrlNotice(url=url, instructions=(
                "Sign in to OpenRouter in your browser and approve the key. If the browser is on another "
                "machine, paste the final redirect URL back here.")))
            prompt = ManualCodePrompt(field_id="return", label="Paste the redirect URL or code here (or wait for the browser)",
                                      accepted="the full redirect URL, or the code")
            returned = await_return(ctx, listener, prompt, lambda pasted: parse_manual_return(
                pasted, expected_state=None, allow_bare_code=True, registered_path=path))
            ctx.notify(ProgressNotice(stage="exchange", message="Exchanging the code for an API key…"))
            return LoginResult(material=_exchange(ctx, returned.code, pkce.verifier), label="OpenRouter (minted key)",
                               renewal="none")
        finally:
            listener.stop()

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        return LoginResult(material=dict(material), label="OpenRouter (minted key)", renewal="none")

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        return RequestAuth(credential=ApiKey(material["key"]))

    def expiry(self, material: Material) -> int | str | None:
        return "never"
