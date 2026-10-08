"""
lm15.login.bound — model choices and the bound client (AUTH-23).

A :class:`BoundClient` is what ``connect()`` returns: one connection id,
one generation, one route, one model.  It follows that connection's
credential renewals and nothing else — a replacement or logout makes it
fail ``connection_changed`` / ``login_required`` instead of quietly
switching who pays (R4).  It builds ordinary canonical ``Request`` values
and returns ordinary ``Response`` / stream events through an ordinary
router; it keeps no conversation, runs no tool loop, retries nothing.

Model choices are evidence-bearing (AUTH-23): each says where it came from
(``application`` = a ``ModelRegistry`` the caller supplied, ``provider`` =
the account's own list fetched now, ``manual`` = an id the caller typed)
and, for a requested capability, whether that is ``supported``,
``unsupported`` or ``unknown``.  Nothing here ranks models, probes with a
paid prompt, or enables a provider policy.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

from ..errors import AuthOperationError
from ..types import Config, Message, Request, Response, StreamEvent, Tool
from .manager import Auth
from .types import Connection, ModelChoice, ModelSelection

__all__ = ["BoundClient", "model_choices", "pinned_auth"]

_CAPABILITIES = {"reasoning", "vision", "structured-output"}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _capability(info: Any, name: str) -> str:
    inference = getattr(info, "inference", None)
    if inference is None:
        return "unknown"
    if name == "reasoning":
        return "supported" if inference.supports_reasoning else "unsupported"
    if name == "vision":
        return "supported" if "image" in inference.input_modalities else "unsupported"
    return "unknown"  # structured output is not recorded in ModelInfo today; say so


def model_choices(
    auth: Auth,
    provider: str,
    *,
    refresh: bool = False,
    capability: str | None = None,
    include_unknown: bool = False,
    registry: Any = None,
    router_config: Any = None,
) -> tuple[ModelChoice, ...]:
    """The models the saved connection on ``provider`` can select.

    ``refresh=False`` reads only a caller-supplied ``ModelRegistry``
    (``application`` source); ``refresh=True`` fetches the account's own
    list with the saved credential (``provider`` source; renews if due; no
    inference).  With ``capability``, only ``supported`` choices are
    returned unless ``include_unknown`` is set (AUTH-23).
    """
    if capability is not None and capability not in _CAPABILITIES:
        raise ValueError(f"capability must be one of {sorted(_CAPABILITIES)}")
    status = auth.status(provider)
    connection = status.connection
    if connection is None:
        raise AuthOperationError(
            f"{provider}: no saved connection to list models for", reason="login_required", stage="catalog",
            recovery="restart_login", provider=provider,
        )
    infos: list[Any] = []
    source = "application"
    fetched: str | None = None
    if refresh:
        from ..router import LMRouter, RouterConfig

        config = replace(router_config or RouterConfig(), auth=auth)
        router = LMRouter(config)
        try:
            lm = router.lm(f"{connection.provider}:catalog")
            infos = list(lm.list_models())
        finally:
            router.close()
        source, fetched = "provider", _now_iso()
    elif registry is not None:
        infos = list(registry.list(connection.provider))
    choices: list[ModelChoice] = []
    for info in infos:
        capabilities = {capability: _capability(info, capability)} if capability else {}
        choice = ModelChoice(provider=connection.provider, model=info.id, connection_id=connection.id, source=source,
                             label=info.id, fetched_at=fetched, capabilities=capabilities)
        if capability and capabilities[capability] == "unsupported":
            continue
        if capability and capabilities[capability] == "unknown" and not include_unknown:
            continue
        choices.append(choice)
    return tuple(choices)


class _PinnedAuth(Auth):
    """The same scope, with every request-time resolution checked against
    one ``(connection_id, generation)``.  Shares the store; owns nothing."""

    def __init__(self, base: Auth, selection: ModelSelection) -> None:
        self.store = base.store
        self._clock = base._clock
        self._monotonic = base._monotonic
        self._opener = base._opener
        self._sleep = base._sleep
        self._active = base._active
        self._lock = base._lock
        self._closed = False
        self._pin = (selection.connection_id, selection.identity_generation)

    def request_auth(self, provider: str, *, pinned: tuple[str, str] | None = None):  # type: ignore[override]
        return super().request_auth(provider, pinned=self._pin)

    def credential_provider(self, provider: str, *, pinned: tuple[str, str] | None = None):  # type: ignore[override]
        return super().credential_provider(provider, pinned=self._pin)


def pinned_auth(auth: Auth, selection: ModelSelection) -> Auth:
    return _PinnedAuth(auth, selection)


class BoundClient:
    """One connection, one model; canonical requests and responses."""

    def __init__(self, auth: Auth, selection: ModelSelection, *, router_config: Any = None) -> None:
        from ..router import LMRouter, RouterConfig

        self.auth = auth
        self.selection = selection
        base = router_config if router_config is not None else RouterConfig()
        if base.auth is not None and base.auth is not auth:
            raise ValueError("router_config.auth must be this client's Auth or None")
        self._router = LMRouter(replace(base, auth=pinned_auth(auth, selection)))

    def __repr__(self) -> str:
        return f"BoundClient({self.selection.routed!r}, connection={self.selection.connection_id!r})"

    @property
    def provider(self) -> str:
        return self.selection.provider

    @property
    def model(self) -> str:
        return self.selection.model

    @property
    def connection(self) -> Connection | None:
        return self.auth.status(self.selection.provider).connection

    # ─── requests ─────────────────────────────────────────────────────

    def request(self, messages: Iterable[Message] | str, *, tools: Sequence[Tool] | None = None,
                config: Config | None = None, system: str | None = None, **fields: Any) -> Request:
        """An ordinary canonical Request with the selected routed model.
        ``messages`` may be a single user string for the short path."""
        if isinstance(messages, str):
            messages = (Message.user(messages),)
        kwargs: dict[str, Any] = {"model": self.selection.routed, "messages": tuple(messages)}
        if tools:
            kwargs["tools"] = tuple(tools)
        if config is not None:
            kwargs["config"] = config
        if system is not None:
            kwargs["system"] = system
        kwargs.update(fields)
        return Request(**kwargs)

    def _coerce(self, request: Request | None, messages: Any, kwargs: dict[str, Any]) -> Request:
        if request is not None:
            if messages is not None or kwargs:
                raise AuthOperationError(
                    "pass either a Request or messages/keywords, not both", reason="selection_mismatch",
                    stage="dispatch", recovery="none", provider=self.provider,
                )
            if request.model not in (self.selection.routed, self.selection.model):
                raise AuthOperationError(
                    f"this client is bound to {self.selection.routed!r}; the Request names {request.model!r}",
                    reason="selection_mismatch", stage="dispatch", recovery="none", provider=self.provider,
                )
            return replace(request, model=self.selection.routed) if request.model != self.selection.routed else request
        if messages is None:
            raise TypeError("complete()/stream() need a Request or messages=")
        return self.request(messages, **kwargs)

    def complete(self, request: Request | None = None, *, messages: Any = None, **kwargs: Any) -> Response:
        return self._router.complete(self._coerce(request, messages, kwargs))

    def stream(self, request: Request | None = None, *, messages: Any = None, **kwargs: Any) -> Iterator[StreamEvent]:
        return self._router.stream(self._coerce(request, messages, kwargs))

    def plan(self, request: Request | None = None, *, messages: Any = None, **kwargs: Any) -> Any:
        return self._router.plan(self._coerce(request, messages, kwargs))

    # ─── lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Release this client's own transport.  Not a logout; the Auth
        stays the caller's."""
        self._router.close()

    def __enter__(self) -> "BoundClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
