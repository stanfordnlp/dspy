"""The vet shim's ``managed_run`` op (lm15-contract harness/PROTOCOL.md § managed).

Runs one scripted program against the public managed-auth API with every
seam injected: the store file the harness created, a fake wall and
monotonic clock (waits advance them), a scripted auth server behind
``Auth(opener=...)`` and a scripted UI. Returns one outcome per step, the
ordered trace and the store file afterwards. The harness compares; this
module only reports.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

from .errors import AuthOperationError, LM15Error
from .login import Auth
from .login.engine import LoginCancelled
from .login.types import AuthUrlNotice, DeviceCodeNotice, InfoNotice, ManualCodePrompt, ProgressNotice, \
    SecretPrompt, SelectPrompt, TextPrompt

JsonObject = dict[str, Any]


class _Clock:
    def __init__(self, start_ms: int) -> None:
        self.start_s = start_ms / 1000.0
        self.elapsed_s = 0.0

    def wall(self) -> float:
        return self.start_s + self.elapsed_s

    def mono(self) -> float:
        return self.elapsed_s


class _Reply:
    def __init__(self, status: int, body: bytes, content_type: str) -> None:
        self.status = status
        self._body = body
        self.headers = {"Content-Type": content_type} if content_type else {}

    def read(self, _n: int = -1) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_Reply":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


def _request_body(request: Any) -> Any:
    data = request.data
    if data is None:
        return None
    text = data.decode("utf-8")
    content_type = (request.get_header("Content-type") or "").split(";")[0].strip()
    if content_type == "application/x-www-form-urlencoded":
        return dict(urllib.parse.parse_qsl(text, keep_blank_values=True))
    try:
        return json.loads(text)
    except ValueError:
        return text


_TRANSPORT_HEADERS = {"accept", "accept-encoding", "connection", "content-length", "content-type", "host"}


def _headers(request: Any) -> JsonObject:
    """The request's own headers, lowercased; transport noise dropped and
    lm15's own User-Agent reduced to its product token (versions differ
    per language)."""
    out: JsonObject = {}
    for name, value in request.header_items():
        key = name.lower()
        if key in _TRANSPORT_HEADERS:
            continue
        if key == "user-agent" and value.startswith("lm15/"):
            value = "lm15"
        out[key] = value
    return out


class _Server:
    def __init__(self, script: list[JsonObject], events: list[JsonObject]) -> None:
        self.script = list(script)
        self.events = events

    def __call__(self, request: Any, _timeout: float) -> _Reply:
        content_type = (request.get_header("Content-type") or "").split(";")[0].strip() or None
        self.events.append({"http": {"method": request.get_method(), "url": request.full_url,
                                     "content_type": content_type, "headers": _headers(request),
                                     "body": _request_body(request)}})
        if not self.script:
            raise urllib.error.URLError(ConnectionRefusedError("no scripted reply"))
        reply = self.script.pop(0)
        if reply.get("delay_ms"):
            import time as _time

            _time.sleep(float(reply["delay_ms"]) / 1000.0)  # real time: lets another process race this exchange
        network = reply.get("network")
        if network == "timeout":
            raise urllib.error.URLError(TimeoutError("timed out"))
        if network == "refused":
            raise urllib.error.URLError(ConnectionRefusedError("refused"))
        if "json" in reply:
            return _Reply(int(reply.get("status", 200)), json.dumps(reply["json"]).encode(), "application/json")
        return _Reply(int(reply.get("status", 200)), str(reply.get("text", "")).encode(),
                      str(reply.get("content_type", "text/plain")))


def _prompt_event(prompt: Any) -> JsonObject:
    event: JsonObject = {"type": prompt.type, "field_id": prompt.field_id}
    if isinstance(prompt, SelectPrompt):
        event["options"] = [option.id for option in prompt.options]
    return event


def _notice_event(notice: Any) -> JsonObject:
    if isinstance(notice, AuthUrlNotice):
        return {"type": "auth_url", "url": notice.url}
    if isinstance(notice, DeviceCodeNotice):
        return {"type": "device_code", "user_code": notice.user_code, "verification_url": notice.verification_url,
                "expires_in_s": notice.expires_in_s, "interval_s": notice.interval_s}
    if isinstance(notice, ProgressNotice):
        return {"type": "progress", "stage": notice.stage}
    if isinstance(notice, InfoNotice):
        return {"type": "info"}
    return {"type": type(notice).__name__}


class _UI:
    def __init__(self, answers: list[Any], events: list[JsonObject]) -> None:
        self.answers = list(answers)
        self.events = events
        self.last_auth_url: str | None = None

    def notify(self, notice: Any) -> None:
        if isinstance(notice, AuthUrlNotice):
            self.last_auth_url = notice.url
        self.events.append({"notice": _notice_event(notice)})

    def prompt(self, prompt: Any) -> str:
        self.events.append({"prompt": _prompt_event(prompt)})
        if not self.answers:
            raise EOFError("the script has no more answers")
        answer = self.answers.pop(0)
        if isinstance(answer, str):
            return answer
        if answer.get("cancel"):
            raise EOFError("the script cancels here")
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.last_auth_url or "").query))
        state = query.get("state", "")
        if "paste" in answer:
            return f"{answer['paste']}#{state}"
        if "paste_wrong_state" in answer:
            return f"{answer['paste_wrong_state']}#not-the-state-of-this-attempt"
        if "paste_url" in answer:
            redirect = query.get("redirect_uri", "")
            return f"{redirect}?{urllib.parse.urlencode({'code': answer['paste_url'], 'state': state})}"
        raise ValueError(f"unknown scripted answer {answer!r}")

    def dismiss(self, _prompt: Any) -> None:
        return None


def _error(exc: BaseException) -> JsonObject:
    if isinstance(exc, LoginCancelled):
        return {"type": "cancelled"}
    if isinstance(exc, AuthOperationError):
        return {"type": "AuthOperationError", "code": exc.code, "reason": exc.reason, "stage": exc.stage,
                "commit_state": exc.commit_state, "recovery": exc.recovery}
    if isinstance(exc, LM15Error):
        return {"type": type(exc).__name__, "code": exc.code}
    return {"type": type(exc).__name__}


def _connection(c: Any) -> JsonObject | None:
    if c is None:
        return None
    out: JsonObject = {
        "id": c.id, "provider": c.provider, "instance_id": c.instance_id, "kind": c.kind, "method_id": c.method_id,
        "routes": list(c.routes), "label": c.label, "created_at": c.created_at,
        "identity_generation": c.identity_generation, "credential_revision": c.credential_revision,
        "settings": dict(c.settings),
    }
    if c.account_label is not None:
        out["account_label"] = c.account_label
    return out


def _status(s: Any) -> JsonObject:
    return {"provider": s.provider, "presence": s.presence, "usability": s.usability,
            "connection": _connection(s.connection), "expires_at": s.expires_at, "logged_out": s.logged_out,
            "verification": None if s.verification is None else {"result": s.verification.result,
                                                                   "check": s.verification.check}}


def _credential(value: Any) -> JsonObject | None:
    from .credentials import ApiKey, BearerToken

    if value is None:
        return None
    if isinstance(value, BearerToken):
        return {"kind": "bearer", "value": value.value}
    if isinstance(value, ApiKey):
        return {"kind": "api_key", "value": value.value}
    if isinstance(value, str):
        return {"kind": "api_key", "value": value}
    return {"kind": type(value).__name__}


def _method(m: Any) -> JsonObject:
    return {"id": m.id, "kind": m.kind, "flow": m.flow, "availability": m.availability,
            "subscription": m.subscription, "delivery": list(m.delivery),
            "fields": [{"id": f.id, "type": f.type, "required": f.required,
                        "options": [o.id for o in f.options]} for f in m.fields]}


def _resolve_refs(step: JsonObject, outcomes: list[JsonObject]) -> JsonObject:
    """``{"id_of_step": n}`` is the connection id step n returned;
    ``{"of_step": n}`` is its ``[id, identity_generation]`` pin."""
    def value(n: int) -> JsonObject:
        outcome = outcomes[n]
        if not outcome.get("ok") or not isinstance(outcome.get("value"), dict):
            raise ValueError(f"step {n} returned no connection to refer to")
        return outcome["value"]

    resolved: JsonObject = {}
    for key, item in step.items():
        if isinstance(item, dict) and "id_of_step" in item:
            resolved[key] = value(int(item["id_of_step"]))["id"]
        elif isinstance(item, dict) and "of_step" in item:
            connection = value(int(item["of_step"]))
            resolved[key] = [connection["id"], connection["identity_generation"]]
        else:
            resolved[key] = item
    return resolved


def _step(auth: Auth, step: JsonObject, clock: _Clock, ui: _UI, env: dict[str, str], sentinel: str) -> Any:
    do = step["do"]
    if do == "advance":
        clock.elapsed_s += float(step["ms"]) / 1000.0
        return None
    if do == "login":
        return _connection(auth.login(
            step["provider"], step.get("method"), ui=ui, answers=step.get("answers"), settings=step.get("settings"),
            replace=step.get("replace"), allow_unverified=bool(step.get("allow_unverified")),
        ))
    if do == "configure":
        return _connection(auth.configure(step["provider"], method=step["method"], answers=step.get("answers"),
                                          settings=step.get("settings"), replace=step.get("replace")))
    if do == "set_api_key":
        return _connection(auth.set_api_key(step["provider"], step["key"], replace=step.get("replace")))
    if do == "status":
        return _status(auth.status(step["provider"]))
    if do == "connections":
        return [_connection(c) for c in auth.connections()]
    if do == "logout":
        forgotten = auth.logout(step["target"])
        return {"provider": forgotten.provider, "forgot": forgotten.forgot, "routes": list(forgotten.routes),
                "identity_generation": forgotten.identity_generation}
    if do == "cancel_login":
        return auth.cancel_login(step["provider"])
    if do == "request_auth":
        pinned = step.get("pinned")
        result = auth.request_auth(step["provider"], pinned=tuple(pinned) if pinned else None)
        return {"credential": _credential(result.credential), "headers": dict(result.headers),
                "base_url": result.base_url, "account_id": result.account_id, "named": result.named}
    if do == "methods":
        return [_method(m) for m in auth.methods(step["provider"])]
    if do == "providers":
        return sorted(d.id for d in auth.providers())
    if do == "explain":
        from .doctor import explain_auth
        from .router import RouterConfig

        api_keys = {p: f"{sentinel}-explicit" for p in step.get("api_keys", [])}
        report = explain_auth(step["provider"], env=env, api_keys=api_keys or None,
                              config=RouterConfig(auth=auth, env=env, api_keys=api_keys or None))
        return {"configured": report.configured, "steps": [{"kind": s.kind, "state": s.state} for s in report.steps]}
    raise ValueError(f"unknown managed step {do!r}")


def op_managed_run(msg: JsonObject) -> JsonObject:
    events: list[JsonObject] = []
    clock = _Clock(int(msg["clock_ms"]))
    env = {str(k): str(v) for k, v in (msg.get("env") or {}).items()}

    def sleep(seconds: float) -> None:
        clock.elapsed_s += seconds
        events.append({"sleep_ms": int(round(seconds * 1000))})

    server = _Server(msg.get("http") or [], events)
    ui = _UI(msg.get("ui") or [], events)
    store_path = Path(msg["store_path"])
    saved_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    steps: list[JsonObject] = []
    try:
        from .login import FileStore

        auth = Auth(FileStore(store_path), clock=clock.wall, monotonic=clock.mono, opener=server, sleep=sleep)
        for index, step in enumerate(msg["steps"]):
            events.append({"step": index})
            try:
                steps.append({"ok": True, "value": _step(auth, _resolve_refs(step, steps), clock, ui, env, str(msg["sentinel"]))})
            except (LM15Error, LoginCancelled, ValueError, KeyError, TypeError) as exc:
                steps.append({"ok": False, "error": _error(exc)})
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
    store: Any = None
    if store_path.exists():
        text = store_path.read_text(encoding="utf-8")
        try:
            store = {"document": json.loads(text)}
        except ValueError:
            store = {"raw": text}
    return {"steps": steps, "events": events, "store": store}
