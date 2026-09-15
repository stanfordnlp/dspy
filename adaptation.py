"""
MAP-13 — adapt freely, never invisibly; refuse only when a guess could hurt.

lm15 makes two promises, ranked: first, change the model or provider
string and the program keeps working; second, never change what the
caller asked for.  The second is kept by VISIBILITY, not refusal.  When a
wire cannot take a setting as asked, the adapter does the obvious thing
and records it here; the record rides ``Response.adaptations`` and
``StreamStartEvent.adaptations``, and ``lm.plan(request)`` previews it
with no network.

Translations — the adapter's ordinary job (``stop`` → ``stop_sequences``,
effort → budget by the MAP-7 table) — are never recorded.  A record exists
only where the wire got something other than what was asked.

The policy (``AdaptationPolicy``) is set on the LM (``adaptations=``) and
on ``RouterConfig``:

- ``"note"`` (default): adapt and record.
- ``"silent"``: adapt and record nothing.
- ``"refuse"``: every DEVIATION — ``dropped``, ``clamped``, ``substituted``,
  ``client_side`` — is an ``UnsupportedFeatureError`` before the wire (the
  pre-2026-09-14 behaviour), carrying ``feature`` = the config path so a
  policy layer can act without parsing prose.  ``satisfied`` and
  ``defaulted`` change nothing the caller asked for (the provider already
  does it; the wire needed a value the caller left open) and are recorded
  under every policy but ``"silent"``.

Nothing here prints.  The record is data on the response; a nagging
warning channel gets ignored, then switched off, and the drop is silent
again.

How the record reaches the response without threading a parameter
through every builder: ``collecting(policy)`` opens a scope (a context
variable — thread-safe, task-safe, and copied into ``asyncio.to_thread``)
around one ``build_request`` call; ``adapt(...)`` inside a builder appends
to the open scope.  ``BaseProviderLM._build`` is the only opener.  A
builder called with no scope open (a test, the vet harness's request
direction) adapts under ``"note"`` and the record is simply not kept.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, get_args

from .errors import UnsupportedFeatureError

AdaptationAction = Literal["dropped", "clamped", "substituted", "client_side", "satisfied", "defaulted"]
ADAPTATION_ACTIONS: tuple[str, ...] = get_args(AdaptationAction)

# The actions that change what the caller asked for; "refuse" refuses these.
DEVIATIONS: frozenset[str] = frozenset({"dropped", "clamped", "substituted", "client_side"})

AdaptationPolicy = Literal["note", "silent", "refuse"]
ADAPTATION_POLICIES: tuple[str, ...] = get_args(AdaptationPolicy)


@dataclass(frozen=True, slots=True)
class Adaptation:
    """One thing the wire got that differs from what was asked.

    - ``field``: the config path (``config.seed``, ``config.temperature``,
      ``config.reasoning.summary``, ``tools``).
    - ``action``: ``dropped`` (a hint with no home), ``clamped`` (a dial
      to its nearest level), ``substituted`` (the closest spelling),
      ``client_side`` (lm15 does it after the wire), ``satisfied`` (the
      provider's default already is what was asked), ``defaulted`` (the
      wire requires a value the caller did not set).
    - ``asked`` / ``applied``: what the caller set / what went to the wire.
      ``asked`` is absent for ``defaulted``; ``applied`` is absent for
      ``dropped`` and ``satisfied``.
    - ``reason``: one sentence naming the provider fact.
    """

    field: str
    action: AdaptationAction
    reason: str
    asked: Any = None
    applied: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("Adaptation.field must be a non-empty string")
        if self.action not in ADAPTATION_ACTIONS:
            raise ValueError(f"unsupported adaptation action: {self.action!r}")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Adaptation.reason must be a non-empty string")


def check_policy(value: object) -> AdaptationPolicy:
    if value not in ADAPTATION_POLICIES:
        raise ValueError(
            f"adaptations must be one of {', '.join(map(repr, ADAPTATION_POLICIES))}, got {value!r}"
        )
    return value  # type: ignore[return-value]


@dataclass(slots=True)
class AdaptationScope:
    policy: AdaptationPolicy
    provider: str | None
    records: list[Adaptation] = field(default_factory=list)
    # plan(): the wire request is built and discarded, so no credential is
    # invoked and none is needed — like resolve(), plan() is offline.
    planning: bool = False


_scope: ContextVar[AdaptationScope | None] = ContextVar("lm15_adaptation_scope", default=None)


@contextmanager
def collecting(policy: AdaptationPolicy, *, provider: str | None = None, planning: bool = False) -> Iterator[AdaptationScope]:
    """Open a scope for one request build.  Nested scopes are independent."""
    scope = AdaptationScope(policy=check_policy(policy), provider=provider, planning=planning)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def adapt(
    field: str,
    action: AdaptationAction,
    reason: str,
    *,
    asked: Any = None,
    applied: Any = None,
    provider: str | None = None,
) -> None:
    """Record one adaptation in the open scope, or raise under ``"refuse"``.

    Builders call this at the point where they would have raised before
    MAP-13, then do the adapted thing.  The message under ``"refuse"`` is
    the same sentence the note carries, so the two policies never say
    different things about the same fact.
    """
    scope = _scope.get()
    policy: AdaptationPolicy = scope.policy if scope is not None else "note"
    who = provider or (scope.provider if scope is not None else None)
    if policy == "refuse" and action in DEVIATIONS:
        head = f"{who}: " if who else ""
        raise UnsupportedFeatureError(
            f"{head}{field} {_ACTION_VERB[action]}: {reason} (adaptations='refuse')",
            provider=who,
            feature=field,
        )
    if policy == "silent" or scope is None:
        return
    scope.records.append(Adaptation(field=field, action=action, reason=reason, asked=asked, applied=applied))


_ACTION_VERB = {
    "dropped": "would be dropped",
    "clamped": "would be clamped",
    "substituted": "would be substituted",
    "client_side": "would be applied client-side",
    "satisfied": "is already satisfied here",
    "defaulted": "would be defaulted",
}


def current_policy() -> AdaptationPolicy:
    scope = _scope.get()
    return scope.policy if scope is not None else "note"


def is_planning() -> bool:
    """True inside ``plan()``: the build's bytes are discarded, so signing
    and credential providers are skipped."""
    scope = _scope.get()
    return scope is not None and scope.planning


def adaptation_to_dict(a: Adaptation) -> dict[str, Any]:
    out: dict[str, Any] = {"field": a.field, "action": a.action, "reason": a.reason}
    if a.asked is not None:
        out["asked"] = a.asked
    if a.applied is not None:
        out["applied"] = a.applied
    return out


def adaptation_from_dict(d: dict[str, Any]) -> Adaptation:
    return Adaptation(
        field=d["field"],
        action=d["action"],
        reason=d["reason"],
        asked=d.get("asked"),
        applied=d.get("applied"),
    )


# ─── Shared clamps ───────────────────────────────────────────────────

EFFORT_LADDER: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")


def nearest_effort(asked: str, available: "tuple[str, ...] | list[str]") -> str:
    """The closest level to ``asked`` among ``available`` on the ordinal
    effort ladder.  A tie goes to the lower level: the cheaper guess is
    the one a caller who set a dial would rather see recorded."""
    levels = [lvl for lvl in available if lvl in EFFORT_LADDER]
    if not levels:
        raise ValueError(f"no comparable effort levels in {available!r}")
    if asked in levels:
        return asked
    want = EFFORT_LADDER.index(asked) if asked in EFFORT_LADDER else 0
    return min(levels, key=lambda lvl: (abs(EFFORT_LADDER.index(lvl) - want), EFFORT_LADDER.index(lvl)))
