"""
lm15.login.manager — ``Auth``: one scope's connections, and their lifecycle.

spec/auth.md AUTH-14 (construction is inert; scope is explicit), AUTH-17
(what each operation may touch), AUTH-19 (generations, replacement, logout,
cancellation ordered against commit), AUTH-20 (renewal under the lock with a
durable in-flight marker; uncertainty is never retried blind), AUTH-24 (typed
outcomes).

Slots.  A slot is one provider route in this scope (v1: instance is always
``public``; a provider-specific setting such as a GHE domain travels on the
connection).  Each slot record in the store's ``_lm15.slots`` carries an
*identity generation* (bumped on every new connection and on logout, never
reused) and a *credential revision* (bumped on every renewal).  A bound
client pins ``(connection_id, generation)``; a general managed router
consults the slot's current connection on every request.

Legacy entries.  A provider entry without a slot record (an xAI login made
by ``lm15.auth.login_xai`` before 2026-09-22, or by Pi) is read as a
connection with generation ``1`` and a stable id derived from the
provider; the record is written on the first managed commit that touches
the slot, so a file that was never touched by the manager is never
rewritten.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..errors import AuthError, AuthOperationError, LM15Error, RateLimitError, ServerError, TransportError
from .engine import ATTEMPT_LIFETIME_S, LoginCancelled, LoginContext, LoginDenied, LoginExpired
from .flows import Material, RequestAuth, descriptor as _descriptor, flow as _flow, flow_for_material, provider_ids
from .store import META_KEY, STORE_VERSION, FileStore, MemoryStore, Store
from .types import AuthUI, Connection, ConnectionStatus, LoginMethod, ProviderDescriptor, SecretPrompt, SelectOption, \
    SelectPrompt, TextPrompt, Verification

__all__ = ["Auth", "AsyncAuth", "ForgetResult", "RENEWAL_LEAD_S"]

RENEWAL_LEAD_S = 300.0  # AUTH-20.3: min(300 s, lifetime / 10)


def _now_iso(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(clock(), tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ForgetResult:
    provider: str
    forgot: bool
    routes: tuple[str, ...]
    identity_generation: str


@dataclass
class _Slot:
    """A slot record as the manager reasons about it (non-secret)."""

    provider: str
    generation: int = 0
    connection_id: str | None = None
    revision: int = 0
    kind: str = "account"
    method_id: str = ""
    instance_id: str = "public"
    label: str = ""
    account_label: str | None = None
    created_at: str = ""
    routes: tuple[str, ...] = ()
    settings: dict[str, str] = field(default_factory=dict)
    state: str = "ready"  # ready | needs_login | indeterminate
    renewal: str = "refresh_token"
    logged_out: bool = False
    renewal_in_flight: dict[str, Any] | None = None
    attempt: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    previous_ids: tuple[str, ...] = ()  # ids this slot held before (never reused; logout by an old id is a no-op)
    legacy: bool = False  # synthesized from an entry with no record

    @classmethod
    def from_record(cls, provider: str, record: dict[str, Any]) -> "_Slot":
        def s(key: str, default: Any = "") -> Any:
            value = record.get(key, default)
            return value if value is not None else default

        return cls(
            provider=provider,
            generation=int(s("generation", "0")), connection_id=record.get("connection_id"),
            revision=int(s("revision", "0")), kind=s("kind", "account"), method_id=s("method_id"),
            instance_id=s("instance_id", "public"), label=s("label"), account_label=record.get("account_label"),
            created_at=s("created_at"), routes=tuple(s("routes", ())), settings=dict(s("settings", {})),
            state=s("state", "ready"), renewal=s("renewal", "refresh_token"), logged_out=bool(record.get("logged_out")),
            renewal_in_flight=record.get("renewal_in_flight"), attempt=record.get("attempt"),
            verification=record.get("verification"), previous_ids=tuple(s("previous_ids", ())),
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "generation": str(self.generation), "connection_id": self.connection_id, "revision": str(self.revision),
            "kind": self.kind, "method_id": self.method_id, "instance_id": self.instance_id, "label": self.label,
            "created_at": self.created_at, "routes": list(self.routes), "settings": dict(self.settings),
            "state": self.state, "renewal": self.renewal,
        }
        if self.account_label:
            record["account_label"] = self.account_label
        if self.logged_out:
            record["logged_out"] = True
        if self.renewal_in_flight:
            record["renewal_in_flight"] = self.renewal_in_flight
        if self.attempt:
            record["attempt"] = self.attempt
        if self.verification:
            record["verification"] = self.verification
        if self.previous_ids:
            record["previous_ids"] = list(self.previous_ids[-8:])
        return record

    def connection(self) -> Connection | None:
        if not self.connection_id:
            return None
        return Connection(
            id=self.connection_id, provider=self.provider, instance_id=self.instance_id, kind=self.kind,
            method_id=self.method_id, routes=self.routes or (self.provider,), label=self.label or self.provider,
            created_at=self.created_at, identity_generation=str(self.generation),
            credential_revision=str(self.revision), settings=dict(self.settings), account_label=self.account_label,
        )


_LEGACY_METHOD = {"xai": "device", "claude-code": "external:claude-code-cli", "openai-codex": "external:codex-cli"}


class Auth:
    """A scope's connections.  ``Auth.local()`` for the private file,
    ``Auth.memory()`` for a process-lifetime store, ``Auth(store)`` for an
    application-supplied one.  Construction reads nothing (AUTH-14)."""

    def __init__(self, store: Store, *, clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic, opener: Any = None,
                 sleep: Callable[[float], None] | None = None) -> None:
        if not isinstance(store, Store):
            raise TypeError("Auth(store) takes a lm15.login.Store")
        self.store = store
        self._clock = clock
        self._monotonic = monotonic
        self._opener = opener
        self._sleep = sleep
        self._active: dict[str, threading.Event] = {}  # provider -> cancel event of a login in this process
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def local(cls, path: str | os.PathLike[str] | None = None) -> "Auth":
        return cls(FileStore(path))

    @classmethod
    def memory(cls) -> "Auth":
        return cls(MemoryStore())

    def __repr__(self) -> str:
        return f"Auth({self.store!r})"

    # ─── discovery (AUTH-13): definitions only ─────────────────────────

    def providers(self) -> tuple[ProviderDescriptor, ...]:
        return tuple(_descriptor(p) for p in provider_ids())

    def methods(self, provider: str) -> tuple[LoginMethod, ...]:
        return self.descriptor(provider).methods

    def descriptor(self, provider: str) -> ProviderDescriptor:
        try:
            return _descriptor(provider)
        except KeyError:
            raise AuthOperationError(
                f"{provider!r} is not a provider lm15 can connect; see Auth.providers()",
                reason="method_unavailable", stage="discovery", recovery="choose_method",
                provider=provider,
            ) from None

    # ─── store views ───────────────────────────────────────────────────

    def _view(self, document: dict[str, Any], provider: str) -> tuple[_Slot, Material | None]:
        meta = document.get(META_KEY) or {}
        record = (meta.get("slots") or {}).get(provider)
        material = document.get(provider)
        if not isinstance(material, dict):
            material = None
        if record is not None:
            return _Slot.from_record(provider, record), material
        slot = _Slot(provider=provider)
        if material is not None:
            # A legacy entry: read as generation 1 with a stable id.
            kind = "account" if material.get("type") == "oauth" else "api_key"
            slot = _Slot(
                provider=provider, generation=1, connection_id=f"legacy-{provider}", revision=1, kind=kind,
                method_id=_LEGACY_METHOD.get(provider, "api_key"), label=f"{provider} (existing login)",
                routes=(provider,), legacy=True,
                renewal="refresh_token" if material.get("type") == "oauth" else "none",
            )
        return slot, material

    @staticmethod
    def _put(document: dict[str, Any], slot: _Slot, material: Material | None) -> dict[str, Any]:
        meta = document.setdefault(META_KEY, {"version": STORE_VERSION, "slots": {}})
        meta.setdefault("version", STORE_VERSION)
        meta.setdefault("slots", {})[slot.provider] = slot.to_record()
        if material is None:
            document.pop(slot.provider, None)
        else:
            document[slot.provider] = material
        return document

    # ─── inspection (AUTH-17: store reads only) ────────────────────────

    def connections(self) -> tuple[Connection, ...]:
        document = self.store.read()
        providers = set(provider_ids())
        found: list[Connection] = []
        for key in sorted(document):
            if key == META_KEY or key not in providers:
                continue
            slot, _ = self._view(document, key)
            connection = slot.connection()
            if connection is not None:
                found.append(connection)
        for key, record in ((document.get(META_KEY) or {}).get("slots") or {}).items():
            if key in document:
                continue
            slot = _Slot.from_record(key, record)
            connection = slot.connection()
            if connection is not None:
                found.append(connection)
        return tuple(found)

    def status(self, provider: str) -> ConnectionStatus:
        provider = self.descriptor(provider).id
        slot, material = self._view(self.store.read(), provider)
        connection = slot.connection()
        verification = None
        if slot.verification:
            verification = Verification(**slot.verification)
        if connection is None:
            return ConnectionStatus(provider=provider, presence="absent", usability="unknown",
                                    logged_out=slot.logged_out,
                                    detail="signed out; sign in again or pass a key explicitly" if slot.logged_out else None)
        usability, expires, detail = self._usability(slot, material)
        return ConnectionStatus(provider=provider, presence="saved", usability=usability, connection=connection,
                                expires_at=expires, verification=verification, detail=detail)

    def _usability(self, slot: _Slot, material: Material | None) -> tuple[str, str | None, str | None]:
        if slot.state == "needs_login":
            return "needs_login", None, "the provider rejected the saved credential; sign in again"
        if slot.state == "indeterminate" or slot.renewal_in_flight:
            return "indeterminate", None, "a renewal was interrupted; sign in again to be safe"
        if material is None:
            return "needs_login", None, "credential material is missing"
        flow = flow_for_material(slot.provider, material)
        expiry = flow.expiry(material)
        if expiry == "never":
            return "ready", "never", None
        if expiry is None:
            return ("ready", "unknown", None) if material.get("type") == "external" else ("unknown", "unknown", None)
        now_ms = int(self._clock() * 1000)
        if now_ms >= expiry - self._lead_ms(flow.lifetime_s(material)):
            if not self._renewable(slot, material):
                return "needs_login", _iso_ms(expiry), "expired and not renewable"
            return "renewal_due", _iso_ms(expiry), None
        return "ready", _iso_ms(expiry), None

    @staticmethod
    def _lead_ms(lifetime_s: float | None) -> int:
        lead = RENEWAL_LEAD_S if lifetime_s is None else min(RENEWAL_LEAD_S, lifetime_s / 10.0)
        return int(lead * 1000)

    @staticmethod
    def _renewable(slot: _Slot, material: Material) -> bool:
        if slot.renewal in ("none", "recipe"):
            return False
        return bool(material.get("refresh"))

    # ─── login (AUTH-16/17/18/19) ──────────────────────────────────────

    def login(
        self,
        provider: str,
        method: str | LoginMethod | None = None,
        *,
        ui: AuthUI,
        settings: dict[str, str] | None = None,
        answers: dict[str, str] | None = None,
        replace: str | None = None,
        cancel: threading.Event | None = None,
        lifetime_s: float = ATTEMPT_LIFETIME_S,
        allow_unverified: bool = False,
    ) -> Connection:
        """Run one login to completion and save the connection.

        ``method`` is a method id or descriptor; omitted, the UI is asked
        when more than one selectable method remains (AUTH-13.6).  ``replace``
        names the connection id being replaced; without it an occupied slot
        is ``connection_exists``.  ``cancel`` is an event the caller may set
        from another thread; Ctrl-C at a prompt cancels too.  The login is
        saved before this returns; a later failure elsewhere never undoes it.
        """
        self._check_open()
        descriptor = self.descriptor(provider)
        provider = descriptor.id
        if not isinstance(lifetime_s, (int, float)) or lifetime_s <= 0:
            raise ValueError("lifetime_s must be a positive number of seconds")
        chosen = self._choose_method(descriptor, method, ui, allow_unverified=allow_unverified)
        answers = dict(answers or {})
        settings = dict(settings or {})
        cancel = cancel or threading.Event()
        deadline = self._monotonic() + float(lifetime_s)
        ctx = LoginContext(ui=ui, deadline=deadline, cancel=cancel, provider=provider, clock=self._monotonic,
                           wall_clock=self._clock, opener=self._opener, sleep=self._sleep)
        attempt_id = f"at_{secrets.token_urlsafe(16)}"
        # Reservation (AUTH-17/18): storage proven writable, one active
        # attempt per slot, generation observed — before any browser opens.
        self.store.reserve()
        expected = self._reserve(provider, attempt_id, replace, lifetime_s)
        with self._lock:
            self._active[provider] = cancel
        try:
            for field_ in chosen.fields:
                if field_.id not in answers:
                    if not field_.required and field_.type != "select":
                        answers[field_.id] = ctx.prompt(TextPrompt(field_.id, field_.label))
                    elif field_.type == "secret":
                        answers[field_.id] = ctx.prompt(SecretPrompt(field_.id, field_.label))
                    elif field_.type == "select":
                        answers[field_.id] = ctx.prompt(SelectPrompt(field_.id, field_.label, field_.options))
                    else:
                        answers[field_.id] = ctx.prompt(TextPrompt(field_.id, field_.label))
            flow = _flow(provider, chosen.id)
            try:
                result = flow.login(ctx, chosen, settings, answers)
            except LoginCancelled:
                self._release(provider, attempt_id)
                raise
            except LoginExpired as exc:
                self._release(provider, attempt_id)
                raise AuthOperationError(
                    f"{provider}: the sign-in was not completed within {int(lifetime_s // 60)} minutes; start again",
                    reason="login_expired", stage="polling", recovery="restart_login", provider=provider,
                    attempt_id=attempt_id, method_id=chosen.id,
                ) from exc
            except LoginDenied as exc:
                self._release(provider, attempt_id)
                raise AuthOperationError(
                    f"{provider}: {exc}", reason="login_denied", stage=exc.stage, recovery="restart_login",
                    provider=provider, attempt_id=attempt_id, method_id=chosen.id,
                    status=exc.status, provider_code=exc.provider_code,
                ) from None
            except TransportError as exc:
                self._release(provider, attempt_id)
                if getattr(exc, "exchange_uncertain", False):
                    raise AuthOperationError(
                        f"{provider}: the network failed after the authorization code may have been sent; the code "
                        "is one-use, so sign in again rather than retry",
                        reason="indeterminate", stage="exchange", commit_state="not_committed",
                        recovery="restart_login", provider=provider, attempt_id=attempt_id, method_id=chosen.id,
                    ) from exc
                raise
            except LM15Error:
                self._release(provider, attempt_id)
                raise
            return self._commit(provider, attempt_id, expected, chosen, result, settings)
        except BaseException:
            # Any exit without a saved connection ends the attempt — Ctrl-C
            # (a notebook's cancel) while waiting on the provider or at a
            # field prompt included — so its reservation must not outlive it
            # and block the next sign-in. Releasing an attempt that is no
            # longer this one's is a no-op.
            self._release(provider, attempt_id)
            raise
        finally:
            with self._lock:
                self._active.pop(provider, None)

    def _choose_method(self, descriptor: ProviderDescriptor, method: str | LoginMethod | None, ui: AuthUI, *,
                       allow_unverified: bool) -> LoginMethod:
        if isinstance(method, LoginMethod):
            method = method.id
        if method is not None:
            try:
                chosen = descriptor.method(method)
            except KeyError:
                raise AuthOperationError(
                    f"{descriptor.id}: no login method {method!r}; see Auth.methods({descriptor.id!r})",
                    reason="method_unavailable", stage="discovery", recovery="choose_method", provider=descriptor.id,
                ) from None
            if chosen.availability == "unavailable":
                raise AuthOperationError(
                    f"{descriptor.id}: method {method!r} is unavailable: {chosen.reason}",
                    reason="method_unavailable", stage="discovery", recovery="choose_method", provider=descriptor.id,
                    method_id=method,
                )
            if chosen.availability == "unverified" and not allow_unverified:
                raise AuthOperationError(
                    f"{descriptor.id}: method {method!r} has no live receipt yet ({chosen.reason}); pass "
                    "allow_unverified=True to try it knowing that",
                    reason="method_unavailable", stage="discovery", recovery="choose_method", provider=descriptor.id,
                    method_id=method,
                )
            return chosen
        candidates = [m for m in descriptor.methods if m.availability == "supported"
                      or (allow_unverified and m.availability == "unverified")]
        if not candidates:
            raise AuthOperationError(
                f"{descriptor.id}: no selectable login method here",
                reason="method_unavailable", stage="discovery", recovery="choose_method", provider=descriptor.id,
            )
        if len(candidates) == 1:
            return candidates[0]
        options = tuple(SelectOption(m.id, m.label, m.billing_note or m.reason) for m in candidates)
        try:
            answer = ui.prompt(SelectPrompt("method", f"How do you want to connect to {descriptor.label}?", options))
        except (KeyboardInterrupt, EOFError):
            raise LoginCancelled("login cancelled at the method prompt") from None
        for candidate in candidates:
            if candidate.id == answer:
                return candidate
        raise AuthOperationError(
            f"{descriptor.id}: the UI answered {answer!r}, which is not one of the offered method ids",
            reason="invalid_login_state", stage="interaction", recovery="choose_method", provider=descriptor.id,
        )

    def _reserve(self, provider: str, attempt_id: str, replace: str | None, lifetime_s: float) -> int:
        """Reserve the slot's single active attempt; return the generation
        the commit must find."""
        now = self._clock()

        def reserve(document: dict[str, Any]) -> dict[str, Any] | None:
            slot, material = self._view(document, provider)
            pending = slot.attempt
            if pending and pending.get("id") != attempt_id:
                started = float(pending.get("started_at_s", 0))
                budget = float(pending.get("lifetime_s", ATTEMPT_LIFETIME_S))
                if now - started < budget:
                    raise AuthOperationError(
                        f"{provider}: another sign-in is already in progress in this scope; finish it or "
                        "cancel it (Auth.cancel_login)",
                        reason="login_in_progress", stage="reservation", recovery="inspect_attempt",
                        provider=provider, attempt_id=pending.get("id"),
                    )
            if slot.connection_id and not replace:
                raise AuthOperationError(
                    f"{provider}: a connection is already saved ({slot.connection_id}); pass replace=that id "
                    "to replace it, or logout first",
                    reason="connection_exists", stage="reservation", recovery="select_connection",
                    provider=provider, connection_id=slot.connection_id,
                )
            if replace and slot.connection_id != replace:
                raise AuthOperationError(
                    f"{provider}: replace={replace!r} does not name the current connection; select again",
                    reason="connection_changed", stage="reservation", recovery="select_connection",
                    provider=provider, connection_id=slot.connection_id,
                )
            slot.attempt = {"id": attempt_id, "expected_generation": str(slot.generation),
                            "started_at_s": now, "lifetime_s": float(lifetime_s)}
            return self._put(document, slot, material)

        document = self.store.mutate(reserve)
        slot, _ = self._view(document, provider)
        return slot.generation

    def _release(self, provider: str, attempt_id: str) -> None:
        def release(document: dict[str, Any]) -> dict[str, Any] | None:
            slot, material = self._view(document, provider)
            if not slot.attempt or slot.attempt.get("id") != attempt_id:
                return None
            slot.attempt = None
            return self._put(document, slot, material)

        try:
            self.store.mutate(release)
        except LM15Error:
            pass  # releasing a reservation must not mask the real failure

    def _commit(self, provider: str, attempt_id: str, expected: int, method: LoginMethod, result: Any,
                settings: dict[str, str]) -> Connection:
        created = _now_iso(self._clock)
        connection_id = f"cn_{secrets.token_urlsafe(12)}"
        routes = tuple(self.descriptor(provider).routes) or (provider,)

        def commit(document: dict[str, Any]) -> dict[str, Any] | None:
            slot, _material = self._view(document, provider)
            if not slot.attempt or slot.attempt.get("id") != attempt_id:
                raise AuthOperationError(
                    f"{provider}: this sign-in was cancelled before it could be saved",
                    reason="invalid_login_state", stage="persistence", commit_state="not_committed",
                    recovery="restart_login", provider=provider, attempt_id=attempt_id,
                )
            if slot.generation != expected:
                raise AuthOperationError(
                    f"{provider}: the saved connection changed while you were signing in; select again",
                    reason="connection_changed", stage="persistence", commit_state="not_committed",
                    recovery="select_connection", provider=provider, attempt_id=attempt_id,
                )
            merged = {**slot.settings, **settings, **result.settings} if slot.connection_id else {**settings, **result.settings}
            previous = slot.previous_ids + ((slot.connection_id,) if slot.connection_id else ())
            new = _Slot(
                provider=provider, generation=slot.generation + 1, connection_id=connection_id, revision=1,
                kind=method.kind, method_id=method.id, label=result.label, account_label=result.account_label,
                created_at=created, routes=routes, settings=merged, state="ready", renewal=result.renewal,
                previous_ids=previous,
            )
            return self._put(document, new, result.material)

        try:
            document = self.store.mutate(commit)
        except AuthOperationError:
            raise
        except LM15Error as exc:
            # A grant may exist at the provider; nothing usable is returned
            # and nothing else is revoked as compensation (AUTH-19).
            raise AuthOperationError(
                f"{provider}: signed in, but the credential could not be saved ({exc.code}); repair the store and "
                "sign in again",
                reason="storage_unavailable", stage="persistence", commit_state="not_committed",
                recovery="repair_storage", provider=provider, attempt_id=attempt_id,
            ) from exc
        slot, _ = self._view(document, provider)
        connection = slot.connection()
        assert connection is not None
        return connection

    def cancel_login(self, provider: str) -> str:
        """Durably cancel the slot's active attempt.  Returns ``"cancelled"``,
        or ``"complete"`` when a commit already won (undo = logout), or
        ``"none"`` when nothing was pending."""
        provider = self.descriptor(provider).id
        outcome = {"result": "none"}

        def cancel(document: dict[str, Any]) -> dict[str, Any] | None:
            slot, material = self._view(document, provider)
            if not slot.attempt:
                outcome["result"] = "complete" if slot.connection_id else "none"
                return None
            slot.attempt = None
            outcome["result"] = "cancelled"
            return self._put(document, slot, material)

        self.store.mutate(cancel)  # the durable record first (AUTH-19) …
        with self._lock:
            event = self._active.get(provider)
        if event is not None:
            event.set()  # … then the running attempt in this process
        return outcome["result"]

    # ─── setup without a provider round-trip (AUTH-17) ─────────────────

    def set_api_key(self, provider: str, key: str, *, replace: str | None = None) -> Connection:
        """Save a literal key (no interpolation, no verification)."""
        if not isinstance(key, str) or not key.strip():
            raise AuthOperationError("set_api_key: the key is empty", reason="interaction_required",
                                     stage="interaction", recovery="provide_input", provider=provider)
        return self.configure(provider, method="api_key", answers={"key": key}, replace=replace)

    def configure(self, provider: str, *, method: str, answers: dict[str, str] | None = None,
                  settings: dict[str, str] | None = None, replace: str | None = None) -> Connection:
        """Save a recipe connection: ``env`` (use ``$VAR`` at request time),
        ``external:<source>`` (another tool's login, read in place),
        ``cloud`` (a named cloud identity), ``local`` (a keyless server),
        or ``api_key``.  No credential is acquired and nothing is verified."""
        self._check_open()
        descriptor = self.descriptor(provider)
        provider = descriptor.id
        try:
            chosen = descriptor.method(method)
        except KeyError:
            raise AuthOperationError(
                f"{provider}: no setup method {method!r}; see Auth.methods({provider!r})",
                reason="method_unavailable", stage="discovery", recovery="choose_method", provider=provider,
            ) from None
        if chosen.flow not in ("form", "source_recipe"):
            raise AuthOperationError(
                f"{provider}: {method!r} is an interactive login; use Auth.login",
                reason="method_unavailable", stage="discovery", recovery="choose_method", provider=provider,
            )
        answers = dict(answers or {})
        settings = dict(settings or {})
        for field_ in chosen.fields:
            if field_.required and not answers.get(field_.id):
                raise AuthOperationError(
                    f"{provider}: {method!r} needs {field_.id!r}", reason="interaction_required",
                    stage="interaction", recovery="provide_input", provider=provider,
                )
        attempt_id = f"at_{secrets.token_urlsafe(16)}"
        self.store.reserve()
        expected = self._reserve(provider, attempt_id, replace, 60.0)
        ctx = LoginContext(ui=_NoUI(), deadline=self._monotonic() + 60.0, provider=provider,
                           clock=self._monotonic, wall_clock=self._clock, opener=self._opener)
        try:
            result = _flow(provider, method).login(ctx, chosen, settings, answers)
        except LoginDenied as exc:
            self._release(provider, attempt_id)
            raise AuthOperationError(f"{provider}: {exc}", reason="login_denied", stage="interaction",
                                     recovery="provide_input", provider=provider) from None
        except LM15Error:
            self._release(provider, attempt_id)
            raise
        return self._commit(provider, attempt_id, expected, chosen, result, settings)

    # ─── logout (AUTH-19) ──────────────────────────────────────────────

    def logout(self, provider_or_connection: str) -> ForgetResult:
        """Forget the connection locally: material removed, generation
        bumped, pending attempt cancelled, a suppression marker kept so a
        restart cannot fall back to an ambient key (R3).  Never calls a
        provider's revoke endpoint; never touches another tool's file."""
        self._check_open()
        provider, target_id = self._resolve_target(provider_or_connection)
        outcome: dict[str, Any] = {}

        def forget(document: dict[str, Any]) -> dict[str, Any] | None:
            slot, material = self._view(document, provider)
            if target_id is not None and slot.connection_id != target_id:
                outcome.update(forgot=False, generation=slot.generation, routes=slot.routes)
                return None  # idempotent: a newer id occupying the slot is untouched
            if slot.connection_id is None and not slot.attempt:
                outcome.update(forgot=False, generation=slot.generation, routes=slot.routes)
                if slot.legacy or slot.generation == 0:
                    return None
                return None
            if material is not None and material.get("type") == "external":
                material_after: Material | None = None  # the recipe goes; the tool's file is untouched
            else:
                material_after = None
            new = _Slot(
                provider=provider, generation=slot.generation + 1, connection_id=None, revision=0,
                kind=slot.kind, method_id=slot.method_id, routes=slot.routes or (provider,), settings={},
                state="ready", renewal="none", logged_out=True,
                previous_ids=slot.previous_ids + ((slot.connection_id,) if slot.connection_id else ()),
            )
            outcome.update(forgot=True, generation=new.generation, routes=new.routes)
            with self._lock:
                event = self._active.get(provider)
            if event is not None:
                event.set()
            return self._put(document, new, material_after)

        self.store.mutate(forget)
        return ForgetResult(provider=provider, forgot=bool(outcome.get("forgot")),
                            routes=tuple(outcome.get("routes") or (provider,)),
                            identity_generation=str(outcome.get("generation", 0)))

    def _resolve_target(self, provider_or_connection: str) -> tuple[str, str | None]:
        if provider_or_connection.startswith(("cn_", "legacy-")):
            for connection in self.connections():
                if connection.id == provider_or_connection:
                    return connection.provider, connection.id
            document = self.store.read()
            for key, record in ((document.get(META_KEY) or {}).get("slots") or {}).items():
                if provider_or_connection in (record.get("previous_ids") or ()):
                    return key, provider_or_connection  # an old id: logout is a no-op, never a newer id's removal
            raise AuthOperationError(
                "no saved connection has that id", reason="attempt_unavailable", stage="resolution",
                recovery="select_connection",
            )
        return self.descriptor(provider_or_connection).id, None

    # ─── verification (AUTH-17) ────────────────────────────────────────

    def verify(self, provider: str, *, router_config: Any = None) -> Verification:
        """An explicit, non-inference check: resolve (renewing if due) and
        list models on the route.  Not universal (some routes have no safe
        check) and possibly metered by the provider."""
        self._check_open()
        provider = self.descriptor(provider).id
        from ..registry import PROVIDERS

        definition = PROVIDERS.get(provider)
        if definition is None or not definition.supports.models:
            return Verification(result="unverified", check="models", detail="this route has no safe non-inference check")
        from ..router import LMRouter, RouterConfig

        config = router_config or RouterConfig()
        from dataclasses import replace as _replace

        router = LMRouter(_replace(config, auth=self))
        checked = _now_iso(self._clock)
        try:
            lm = router.lm(f"{provider}:verify")
            lm.list_models()
            result = Verification(result="valid", checked_at=checked, check="models")
        except AuthError as exc:
            result = Verification(result="rejected", checked_at=checked, check="models", detail=exc.code)
        finally:
            router.close()
        self._record_verification(provider, result)
        return result

    def _record_verification(self, provider: str, result: Verification) -> None:
        def record(document: dict[str, Any]) -> dict[str, Any] | None:
            slot, material = self._view(document, provider)
            if slot.connection_id is None:
                return None
            slot.verification = {"result": result.result, "checked_at": result.checked_at, "check": result.check,
                                 "detail": result.detail}
            return self._put(document, slot, material)

        try:
            self.store.mutate(record)
        except LM15Error:
            pass

    # ─── request-time resolution (AUTH-15/20) ──────────────────────────

    def request_auth(self, provider: str, *, pinned: tuple[str, str] | None = None) -> RequestAuth:
        """What a request on ``provider`` sends now: the saved connection's
        credential, renewed under the lock if due.  ``pinned`` is a bound
        client's ``(connection_id, generation)``; a mismatch is
        ``connection_changed``, never a silent rebind (AUTH-20.1)."""
        provider = self.descriptor(provider).id
        document = self.store.read()
        slot, material = self._view(document, provider)
        if slot.renewal_in_flight and slot.state != "indeterminate":
            # A sibling may be renewing right now (it holds the lock), or may
            # have died mid-exchange. Only the lock can tell: wait for it,
            # re-read, and reuse the sibling's result; a marker still there
            # once we hold the lock is an interrupted renewal (AUTH-20.4).
            return self._renew(provider, pinned)
        self._check_selected(provider, slot, material, pinned)
        flow = flow_for_material(provider, material)  # type: ignore[arg-type]
        assert material is not None
        expiry = flow.expiry(material)
        if expiry in ("never", None):
            return self._auth_from(provider, flow, material, slot)
        now_ms = int(self._clock() * 1000)
        if now_ms < expiry - self._lead_ms(flow.lifetime_s(material)):
            return self._auth_from(provider, flow, material, slot)
        return self._renew(provider, pinned)

    def _check_selected(self, provider: str, slot: _Slot, material: Material | None, pinned: tuple[str, str] | None) -> None:
        if pinned is not None and (slot.connection_id != pinned[0] or str(slot.generation) != pinned[1]):
            if slot.connection_id is None:
                raise AuthOperationError(
                    f"{provider}: the connection this client was bound to was signed out; connect again",
                    reason="login_required", stage="resolution", recovery="restart_login", provider=provider,
                    connection_id=pinned[0],
                )
            raise AuthOperationError(
                f"{provider}: the saved connection was replaced after this client was bound; connect again",
                reason="connection_changed", stage="resolution", recovery="select_connection", provider=provider,
                connection_id=pinned[0],
            )
        if slot.connection_id is None or material is None:
            if slot.logged_out:
                message = f"{provider}: signed out; sign in again (Auth.login) or pass a key explicitly (api_keys)"
            else:
                message = f"{provider}: no saved connection in this scope; sign in with Auth.login or connect()"
            raise AuthOperationError(message, reason="login_required", stage="resolution",
                                     recovery="restart_login", provider=provider)
        if slot.state == "needs_login":
            raise AuthOperationError(
                f"{provider}: the saved credential was rejected by the provider; sign in again",
                reason="login_required", stage="resolution", recovery="restart_login", provider=provider,
                connection_id=slot.connection_id,
            )
        if slot.state == "indeterminate" or slot.renewal_in_flight:
            raise AuthOperationError(
                f"{provider}: a credential renewal was interrupted and its outcome is unknown; sign in again "
                "rather than reuse a possibly consumed token",
                reason="indeterminate", stage="resolution", commit_state="unknown", recovery="restart_login",
                provider=provider, connection_id=slot.connection_id,
            )

    def _auth_from(self, provider: str, flow: Any, material: Material, slot: _Slot) -> RequestAuth:
        try:
            return flow.request_auth(material, slot.settings)
        except LoginDenied as exc:
            raise AuthOperationError(f"{provider}: {exc}", reason="login_required", stage="resolution",
                                     recovery="restart_login", provider=provider,
                                     connection_id=slot.connection_id) from None

    def _renew(self, provider: str, pinned: tuple[str, str] | None) -> RequestAuth:
        """AUTH-20.4: lock, re-read, reuse a sibling's fresh result, else
        mark in-flight, exchange, write — all under the lock."""
        ctx = LoginContext(ui=_NoUI(), deadline=self._monotonic() + 60.0, provider=provider, clock=self._monotonic,
                           wall_clock=self._clock, opener=self._opener)
        with self.store.transaction() as txn:
            document = txn.read()
            slot, material = self._view(document, provider)
            self._check_selected(provider, slot, material, pinned)
            assert material is not None
            flow = flow_for_material(provider, material)
            expiry = flow.expiry(material)
            now_ms = int(self._clock() * 1000)
            if expiry in ("never", None) or now_ms < expiry - self._lead_ms(flow.lifetime_s(material)):
                return self._auth_from(provider, flow, material, slot)  # a sibling renewed while we waited
            if not self._renewable(slot, material):
                self._mark(txn, document, slot, material, state="needs_login", drop_material=True)
                raise AuthOperationError(
                    f"{provider}: the saved credential expired and cannot be renewed; sign in again",
                    reason="credential_rejected", stage="renewal", commit_state="committed",
                    recovery="restart_login", provider=provider, connection_id=slot.connection_id,
                )
            # Durable in-flight marker before the possibly rotating exchange.
            slot.renewal_in_flight = {"started_at": _now_iso(self._clock), "revision": str(slot.revision)}
            txn.write(self._put(document, slot, material))
            try:
                result = flow.renew(ctx, material, slot.settings)
            except LoginDenied as exc:
                self._mark(txn, document, slot, material, state="needs_login", drop_material=True)
                raise AuthOperationError(
                    f"{provider}: renewal failed ({exc}); sign in again",
                    reason="credential_rejected", stage="renewal", commit_state="committed",
                    recovery="restart_login", provider=provider, connection_id=slot.connection_id,
                    status=exc.status, provider_code=exc.provider_code,
                ) from None
            except (RateLimitError, ServerError) as exc:
                self._mark(txn, document, slot, material, state="ready")  # known safe: keep credentials
                raise exc
            except TransportError as exc:
                if getattr(exc, "exchange_uncertain", False):
                    self._mark(txn, document, slot, material, state="indeterminate", keep_marker=True)
                    raise AuthOperationError(
                        f"{provider}: the renewal exchange timed out after it may have reached the provider; a "
                        "rotated token cannot be spent twice, so sign in again",
                        reason="indeterminate", stage="renewal", commit_state="unknown", recovery="restart_login",
                        provider=provider, connection_id=slot.connection_id,
                    ) from exc
                self._mark(txn, document, slot, material, state="ready")
                raise
            except BaseException:
                self._mark(txn, document, slot, material, state="indeterminate", keep_marker=True)
                raise
            slot.renewal_in_flight = None
            slot.revision += 1
            slot.state = "ready"
            if result.account_label:
                slot.account_label = result.account_label
            txn.write(self._put(document, slot, result.material))
            return self._auth_from(provider, flow, result.material, slot)

    def _mark(self, txn: Any, document: dict[str, Any], slot: _Slot, material: Material | None, *, state: str,
              drop_material: bool = False, keep_marker: bool = False) -> None:
        slot.state = state
        if not keep_marker:
            slot.renewal_in_flight = None
        txn.write(self._put(document, slot, None if drop_material else material))

    def credential_provider(self, provider: str, *, pinned: tuple[str, str] | None = None) -> Callable[[], Any]:
        """A zero-argument callable the adapters resolve per request
        (AUTH-2): each call is :meth:`request_auth`."""
        provider = self.descriptor(provider).id

        def resolve() -> Any:
            return self.request_auth(provider, pinned=pinned).credential

        resolve.__name__ = f"lm15_managed_{provider}"
        return resolve

    # ─── housekeeping ─────────────────────────────────────────────────

    def close(self) -> None:
        """Cancel logins this manager is running; never a logout."""
        with self._lock:
            self._closed = True
            events = list(self._active.values())
        for event in events:
            event.set()

    def _check_open(self) -> None:
        if self._closed:
            raise AuthOperationError("this Auth was closed", reason="storage_unavailable", stage="resolution",
                                     recovery="operator_action")

    def __enter__(self) -> "Auth":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class _NoUI:
    """A UI that cannot answer: any prompt is ``interaction_required``."""

    def prompt(self, prompt: Any) -> str:
        raise AuthOperationError(
            "this operation needs a choice or input and no UI was supplied",
            reason="interaction_required", stage="interaction", recovery="provide_input",
        )

    def notify(self, notice: Any) -> None:
        return None


class AsyncAuth:
    """The same manager for ``async`` code (AUTH-22: Python gets a native
    async entry point).  Logins are long, interactive and I/O-bound, so
    each operation runs the sync manager in a worker thread and awaits it;
    nothing blocks the event loop and nothing pretends to be non-blocking
    while doing blocking I/O.  ``cancel`` events and the store are shared
    with the wrapped :class:`Auth`, which ``RouterConfig(auth=...)`` and
    ``AsyncLMRouter`` take directly (``AsyncAuth.sync``)."""

    def __init__(self, sync: Auth) -> None:
        self.sync = sync

    @classmethod
    def local(cls, path: str | os.PathLike[str] | None = None) -> "AsyncAuth":
        return cls(Auth.local(path))

    @classmethod
    def memory(cls) -> "AsyncAuth":
        return cls(Auth.memory())

    def __repr__(self) -> str:
        return f"AsyncAuth({self.sync.store!r})"

    # definitions only: no thread needed
    def providers(self):
        return self.sync.providers()

    def methods(self, provider: str):
        return self.sync.methods(provider)

    async def _run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        import asyncio
        import functools

        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))

    async def connections(self):
        return await self._run(self.sync.connections)

    async def status(self, provider: str):
        return await self._run(self.sync.status, provider)

    async def login(self, provider: str, method: Any = None, **kwargs: Any):
        return await self._run(self.sync.login, provider, method, **kwargs)

    async def set_api_key(self, provider: str, key: str, **kwargs: Any):
        return await self._run(self.sync.set_api_key, provider, key, **kwargs)

    async def configure(self, provider: str, **kwargs: Any):
        return await self._run(self.sync.configure, provider, **kwargs)

    async def verify(self, provider: str, **kwargs: Any):
        return await self._run(self.sync.verify, provider, **kwargs)

    async def logout(self, provider_or_connection: str):
        return await self._run(self.sync.logout, provider_or_connection)

    async def cancel_login(self, provider: str):
        return await self._run(self.sync.cancel_login, provider)

    async def request_auth(self, provider: str, **kwargs: Any):
        return await self._run(self.sync.request_auth, provider, **kwargs)

    def close(self) -> None:
        self.sync.close()

    async def __aenter__(self) -> "AsyncAuth":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self.close()
