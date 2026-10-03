"""
lm15.login.flows.base — what a provider flow is.

A flow describes one provider's login protocol and nothing else.  The
engine owns deadlines, cancellation, UI, HTTP bounds and device pacing; the
store owns persistence; the manager owns lifecycle.  A flow answers four
questions:

- ``login``: run the chosen method and return :class:`LoginResult`;
- ``renew``: given the saved material, produce fresh material or raise
  :class:`~lm15.login.engine.LoginDenied` (permanent: the manager marks
  ``needs_login``);
- ``request_auth``: derive what a request needs from valid material
  (credential, extra headers, a credential-dependent base URL);
- ``expiry``: read the material's actual expiry.

Material is the provider-private dict saved under the provider's key in
the store.  For OAuth providers it follows the Pi/lm15 convention
``{"type": "oauth", "access", "refresh", "expires"}`` so the entry stays
readable by the legacy loader (xAI) and by Pi; ``expires`` is the actual
expiry in epoch milliseconds (AUTH-20.2), ``issued_at`` its issue time,
and ``lifetime_s`` the granted lifetime — the three numbers the ratified
renewal lead (``min(300 s, lifetime / 10)``) is computed from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..engine import LoginContext
from ..types import LoginMethod, ProviderDescriptor

__all__ = ["ProviderFlow", "LoginResult", "RequestAuth", "oauth_material", "material_expiry_ms"]

Material = dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoginResult:
    material: Material = field(repr=False)
    label: str
    account_label: str | None = None
    renewal: str = "refresh_token"  # refresh_token | remint | none | external | recipe
    settings: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RequestAuth:
    """What the adapter sends.  ``credential`` is a str or an AUTH-2 value."""

    credential: Any = field(repr=False)
    headers: dict[str, str] = field(default_factory=dict)
    base_url: str | None = None
    account_id: str | None = None
    named: str | None = None  # a saved cloud recipe: the named identity to run (AUTH-15)


def oauth_material(*, access: str, refresh: str | None, expires_in_s: float | None, now_ms: int,
                   extra: dict[str, Any] | None = None) -> Material:
    """Build OAuth material with the actual expiry and the numbers the
    renewal lead needs.  ``expires_in_s=None`` records ``unknown``."""
    material: Material = {"type": "oauth", "access": access}
    if refresh:
        material["refresh"] = refresh
    material["issued_at"] = now_ms
    if expires_in_s is not None and expires_in_s > 0:
        material["lifetime_s"] = float(expires_in_s)
        material["expires"] = int(now_ms + expires_in_s * 1000)
    if extra:
        material.update(extra)
    return material


def material_expiry_ms(material: Material) -> int | None:
    value = material.get("expires")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


class ProviderFlow:
    """Subclass per provider.  ``descriptor`` is the AUTH-13 description."""

    descriptor: ProviderDescriptor

    def login(self, ctx: LoginContext, method: LoginMethod, settings: dict[str, str], answers: dict[str, str]) -> LoginResult:
        raise NotImplementedError

    def renew(self, ctx: LoginContext, material: Material, settings: dict[str, str]) -> LoginResult:
        raise NotImplementedError

    def request_auth(self, material: Material, settings: dict[str, str]) -> RequestAuth:
        raise NotImplementedError

    def expiry(self, material: Material) -> int | str | None:
        """Actual expiry in epoch ms, ``"never"``, or ``None`` for unknown."""
        if material.get("type") == "api_key":
            return "never"
        return material_expiry_ms(material)

    def lifetime_s(self, material: Material) -> float | None:
        value = material.get("lifetime_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
        issued, expires = material.get("issued_at"), material_expiry_ms(material)
        if isinstance(issued, (int, float)) and not isinstance(issued, bool) and expires:
            return max((expires - int(issued)) / 1000.0, 0.0)
        return None
