"""
lm15.interactive — ``connect()``: get ready to make model requests.

The explicitly interactive module (spec/auth.md AUTH-23).  Import it when
a person is at a terminal; a server never should — ``connect()`` fails
before reading any secret when there is no terminal and no UI was given.

    from lm15 import Message
    from lm15.interactive import connect

    with connect() as lm:
        answer = lm.complete(messages=[Message.user("Explain drought stress.")])
        print(answer.text)

What it does, in order: shows where connections are saved; offers the
saved connections and "connect another" (subscriptions first — an
explicit key or a named cloud identity is deliberate authority, an
ambient environment key is offered only as an explicit choice, never
taken silently: R2/R3); runs the chosen login or setup; fetches the
account's model list (or takes ``model=``) and asks; returns a
:class:`~lm15.login.bound.BoundClient` pinned to that connection and
model.  A completed login is saved *before* the model picker runs: cancel
the picker and the login stays (R6).

What it never does: send a prompt, set a process-wide default, change
another router, fall back to a different account when one fails.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .errors import AuthOperationError
from .login.bound import BoundClient, model_choices
from .login.engine import LoginCancelled
from .login.manager import Auth
from .login.terminal import TerminalUI
from .login.types import AuthUI, Connection, InfoNotice, LoginMethod, ModelSelection, ProviderDescriptor, \
    SelectOption, SelectPrompt, TextPrompt

__all__ = ["connect"]

_NEW = "__new__"
_MANUAL = "__manual__"
_KEY_FROM_ENV = "__env__"


def _interactive_terminal() -> bool:
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def connect(
    provider: str | None = None,
    *,
    model: str | None = None,
    auth: Auth | None = None,
    ui: AuthUI | None = None,
    capability: str | None = None,
    open_browser: bool = False,
    router_config: Any = None,
    allow_unverified: bool = False,
) -> BoundClient:
    """Choose (or make) a connection and a model; return a bound client.

    ``provider`` and ``model`` skip the corresponding pickers when given.
    ``auth`` defaults to ``Auth.local()`` (the private file); ``ui`` to a
    terminal adapter on an interactive terminal.  ``capability`` filters
    the model picker to models known to support it (``reasoning``,
    ``vision``, ``structured-output``); unknown ones are not offered.
    ``allow_unverified`` lets the picker offer login methods that exist
    but have no live receipt yet, labelled as such.
    """
    if ui is None:
        if not _interactive_terminal():
            raise AuthOperationError(
                "connect() needs a person: no interactive terminal here and no ui= was supplied. On a server, "
                "attach an Auth with saved connections (RouterConfig(auth=...)) instead of calling connect().",
                reason="interaction_required", stage="interaction", recovery="provide_input",
            )
        ui = TerminalUI(open_browser=open_browser)
    auth = auth or Auth.local()
    ui.notify(InfoNotice(f"Connections are saved privately in {auth.store.description}."))

    connection = _choose_connection(auth, ui, provider, allow_unverified=allow_unverified)
    selection = _choose_model(auth, ui, connection, model=model, capability=capability, router_config=router_config)
    ui.notify(InfoNotice(f"Ready: {selection.routed} through {connection.label}."))
    return BoundClient(auth, selection, router_config=router_config)


# ─── connection ──────────────────────────────────────────────────────


def _choose_connection(auth: Auth, ui: AuthUI, provider: str | None, *, allow_unverified: bool) -> Connection:
    saved = [c for c in auth.connections() if provider is None or c.provider == auth.descriptor(provider).id]
    # Subscriptions first (R2): account connections before keys.
    saved.sort(key=lambda c: (0 if c.kind == "account" else 1, c.provider))
    usable = []
    for connection in saved:
        status = auth.status(connection.provider)
        if status.usability in ("ready", "renewal_due", "unknown"):
            usable.append(connection)
    if provider is not None and len(usable) == 1:
        return usable[0]
    options = [SelectOption(c.id, f"{c.label}", f"{c.provider} · saved") for c in usable]
    options.append(SelectOption(_NEW, "Connect another account or API key"))
    if len(options) == 1:
        return _new_connection(auth, ui, provider, allow_unverified=allow_unverified)
    answer = _ask(ui, SelectPrompt("connection", "Use a saved connection, or connect another?", tuple(options)))
    if answer == _NEW:
        return _new_connection(auth, ui, provider, allow_unverified=allow_unverified)
    for connection in usable:
        if connection.id == answer:
            return connection
    raise AuthOperationError("the UI answered with an unknown connection id", reason="invalid_login_state",
                             stage="interaction", recovery="select_connection")


def _new_connection(auth: Auth, ui: AuthUI, provider: str | None, *, allow_unverified: bool) -> Connection:
    if provider is None:
        descriptors = [d for d in auth.providers() if any(m.availability != "unavailable" for m in d.methods)]
        # Subscriptions first, then everything else alphabetically.
        descriptors.sort(key=lambda d: (0 if any(m.subscription and m.availability == "supported" for m in d.methods) else 1,
                                        d.label.lower()))
        options = tuple(SelectOption(d.id, d.label, d.service if d.service != d.label else None) for d in descriptors)
        provider = _ask(ui, SelectPrompt("provider", "Which provider?", options))
    descriptor = auth.descriptor(provider)
    existing = auth.status(descriptor.id).connection
    method = _choose_method(ui, descriptor, allow_unverified=allow_unverified, env=os.environ)
    replace = None
    if existing is not None:
        answer = _ask(ui, SelectPrompt("replace", f"{descriptor.label} already has a saved connection ({existing.label}).",
                                       (SelectOption("keep", "Keep it"), SelectOption("replace", "Replace it"))))
        if answer == "keep":
            return existing
        replace = existing.id
    if method.flow in ("form", "source_recipe"):
        answers: dict[str, str] = {}
        for field_ in method.fields:
            if field_.type == "select" and len(field_.options) == 1:
                answers[field_.id] = field_.options[0].id
            elif field_.type == "select":
                answers[field_.id] = _ask(ui, SelectPrompt(field_.id, field_.label, field_.options))
            elif field_.type == "secret":
                from .login.types import SecretPrompt

                answers[field_.id] = _ask(ui, SecretPrompt(field_.id, field_.label))
            else:
                answers[field_.id] = _ask(ui, TextPrompt(field_.id, field_.label))
        return auth.configure(descriptor.id, method=method.id, answers=answers, replace=replace)
    try:
        return auth.login(descriptor.id, method.id, ui=ui, replace=replace, allow_unverified=allow_unverified)
    except LoginCancelled:
        raise AuthOperationError("sign-in cancelled", reason="interaction_required", stage="interaction",
                                 recovery="restart_login", provider=descriptor.id) from None


def _choose_method(ui: AuthUI, descriptor: ProviderDescriptor, *, allow_unverified: bool, env: Any) -> LoginMethod:
    methods = [m for m in descriptor.methods if m.availability == "supported"
               or (allow_unverified and m.availability == "unverified")]
    if not methods:
        raise AuthOperationError(
            f"{descriptor.id}: no login method is available here", reason="method_unavailable", stage="discovery",
            recovery="choose_method", provider=descriptor.id,
        )
    # Subscriptions first; an ambient key is offered, never assumed (R2).
    methods.sort(key=lambda m: (0 if m.subscription else 1, 0 if m.kind == "account" else 1))
    options = []
    for method in methods:
        note = method.billing_note
        if method.availability == "unverified":
            note = f"UNVERIFIED — {method.reason}"
        if method.id == "env":
            set_names = [o.id for o in method.fields[0].options if env.get(o.id)] if method.fields else []
            if not set_names:
                continue  # nothing to offer
            note = f"${set_names[0]} is set in this environment; using it is your explicit choice"
        options.append(SelectOption(method.id, method.label, note))
    if len(options) == 1:
        chosen = options[0].id
    else:
        chosen = _ask(ui, SelectPrompt("method", f"How do you want to connect to {descriptor.label}?", tuple(options)))
    return descriptor.method(chosen)


# ─── model ───────────────────────────────────────────────────────────


def _choose_model(auth: Auth, ui: AuthUI, connection: Connection, *, model: str | None, capability: str | None,
                  router_config: Any) -> ModelSelection:
    if model is not None:
        return ModelSelection(provider=connection.provider, model=model, connection_id=connection.id,
                              identity_generation=connection.identity_generation)
    choices = ()
    try:
        choices = model_choices(auth, connection.provider, refresh=True, capability=capability, router_config=router_config)
        source_note = "listed by your account just now"
    except AuthOperationError:
        raise
    except Exception as exc:  # the catalog is a convenience; say why it is missing, do not pretend
        ui.notify(InfoNotice(f"Could not list models for {connection.provider} ({type(exc).__name__}); type a model id."))
        source_note = ""
    options = [SelectOption(c.model, c.model, source_note or None) for c in choices]
    options.append(SelectOption(_MANUAL, "Type a model id (not verified against your account)"))
    if capability and not choices:
        ui.notify(InfoNotice(f"No model in the list is known to support {capability!r}; you can still type one."))
    answer = _ask(ui, SelectPrompt("model", f"Which {connection.provider} model?", tuple(options)))
    if answer == _MANUAL:
        answer = _ask(ui, TextPrompt("model", "Model id")).strip()
        if not answer:
            raise AuthOperationError("no model id given", reason="interaction_required", stage="interaction",
                                     recovery="provide_input", provider=connection.provider)
    return ModelSelection(provider=connection.provider, model=answer, connection_id=connection.id,
                          identity_generation=connection.identity_generation)


def _ask(ui: AuthUI, prompt: Any) -> str:
    try:
        return ui.prompt(prompt)
    except (KeyboardInterrupt, EOFError):
        raise AuthOperationError("cancelled", reason="interaction_required", stage="interaction",
                                 recovery="restart_login") from None
