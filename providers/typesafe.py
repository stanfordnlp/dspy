"""TypeSafe System One (Jev) — provider ``typesafe``.

changes/2026-09-17-judgments.md.  One ``POST /v1/systemone`` per Request:
the messages become Jev's ``state`` (D6), the judgment properties of the
``json_schema`` become its ``questions`` (MAP-14 §2), and the answers come
back as one ``DataPart`` with the distribution per judgment and
``method="provider_classification"`` (§3).  Jev generates no text: a
request without judgments, with tools, or with media is refused before
the wire (D8).  Wire facts: receipts/2026-09-17-judgments/ (jev-*.json).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ClassVar, Iterator, Mapping

from ..access import TYPESAFE_API
from ..adaptation import AdaptationPolicy, adapt, check_policy
from ..errors import (
    AuthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServerError,
    UnsupportedFeatureError,
    UnsupportedModelError,
    map_http_error,
)
from ..features import ProviderManifest
from ..judgments import MAX_CHOICE_KEYS, MAX_ORDERED_LEVELS, Judgment, non_judgment_properties, request_judgments
from ..transports import TransportRequest
from ..types import DataPart, Message, Request, Response, StreamEvent, TextPart, Usage
from .base import BaseProviderLM, Credential, HttpResponse, SyncTransport, default_transport, _attach_error_metadata
from .common import model_infos_from_entries
from ..sse import SSEEvent

_DEFAULT_BASE_URL = "https://api.typesafe.ai"

# Config knobs with no home on the systemone wire (D8): dropped with a record.
_DROPPED_KNOBS: tuple[str, ...] = (
    "max_tokens", "temperature", "top_p", "top_k", "stop", "seed", "frequency_penalty",
    "presence_penalty", "reasoning", "logprobs", "store", "user_id", "service_tier", "cache",
)


@dataclass
class TypeSafeLM(BaseProviderLM):
    """TypeSafe System One dialect (``POST /v1/systemone``)."""

    api_key: Credential | None = field(default=None, repr=False)
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = _DEFAULT_BASE_URL
    access: ProviderManifest | None = None
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = None
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False)
    adaptations: AdaptationPolicy = field(default="note", kw_only=True)
    provider: str = field(default="typesafe", init=False)
    account_id: str | None = field(default=None, init=False, repr=False)
    manifest: ClassVar[ProviderManifest] = TYPESAFE_API

    def __post_init__(self) -> None:
        check_policy(self.adaptations)
        self._bind_access(self.access, credentials_path=self.credentials_path,
                          default_base_url=_DEFAULT_BASE_URL, settings=self.settings)

    # ─── Request building (pure) ────────────────────────────────────

    def _refuse(self, feature: str, why: str) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"{self.provider}: {why}", provider=self.provider, feature=feature)

    def _state(self, request: Request) -> Any:
        """changes/2026-09-19-jev-state.md D1/D2: the state is the one user
        part, verbatim — a text's string or a data part's value. Jev has
        no system prompt and no conversation; anything else is refused
        with the native place named, never merged into a shape of ours.
        A media part is refused first, as the specific fault it is (MAP-10)."""
        for m_index, message in enumerate(request.messages):
            for p_index, part in enumerate(message.parts):
                if not isinstance(part, (TextPart, DataPart)):
                    raise self._refuse(
                        f"messages[{m_index}].parts[{p_index}]",
                        f"a {part.type} part has no slot on the systemone wire (MAP-10); Jev reads text or data",
                    )
        if request.system is not None:
            raise self._refuse(
                "system",
                "Jev has no system prompt; put context in the state as a named key "
                "(Message.user(data({\"policy\": ..., \"note\": ...}))), or the framing in each "
                "question's description (changes/2026-09-19-jev-state.md D2)",
            )
        if len(request.messages) != 1:
            raise self._refuse(
                "messages",
                f"Jev judges one state, got {len(request.messages)} messages; put a transcript in the "
                "state as an array or object (Message.user(data({\"messages\": [...]}))), where a "
                "question can point at a turn with a backtick path (changes/2026-09-19-jev-state.md D2)",
            )
        message = request.messages[0]
        if message.role != "user":
            raise self._refuse("messages[0].role", f"Jev's state is a user message, got role {message.role!r}")
        if len(message.parts) != 1:
            raise self._refuse(
                "messages[0].parts",
                f"Jev's state is one text or data part, got {len(message.parts)} parts; put several pieces "
                "in one data part as named keys",
            )
        only = message.parts[0]
        return only.text if isinstance(only, TextPart) else only.value  # type: ignore[union-attr]

    def _questions(self, request: Request) -> dict[str, Any]:
        fmt = request.config.response_format
        if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
            raise self._refuse(
                "config.response_format",
                "Jev answers declared judgments only; give a json_schema response_format whose "
                "properties are enums / booleans / ordered levels (MAP-14), e.g. lm15.judgments(...)",
            )
        found = request_judgments(request)
        extra = non_judgment_properties(fmt.get("schema"), found)
        if not found or extra:
            what = f"properties {list(extra)} are free-form" if extra else "no property declares a judgment"
            raise self._refuse(
                "config.response_format",
                f"{what}; Jev cannot generate values, only pick among declared keys (MAP-14 §1)",
            )
        questions: dict[str, Any] = {}
        for name, j in found.items():
            instruction = j.instruction
            if instruction is None:
                adapt(
                    f"config.response_format.schema.properties.{name}.description",
                    "defaulted",
                    "a judgment without a description: the property name goes as the instruction (Jev never sees property names)",
                    applied=name,
                    provider=self.provider,
                )
                instruction = name
            if j.kind == "boolean":
                questions[name] = {"type": "noul", "instructions": instruction}
            elif j.kind == "choice":
                if len(j.keys) > MAX_CHOICE_KEYS:
                    raise self._refuse(f"config.response_format.schema.properties.{name}",
                                       f"a Jev choice takes at most {MAX_CHOICE_KEYS} keys, got {len(j.keys)}")
                questions[name] = {"type": "choice", "instructions": instruction,
                                   "criteria": {k: j.descriptions.get(k) for k in j.keys}}
            else:
                if len(j.keys) > MAX_ORDERED_LEVELS:
                    raise self._refuse(f"config.response_format.schema.properties.{name}",
                                       f"a Jev score takes at most {MAX_ORDERED_LEVELS} levels, got {len(j.keys)}")
                criteria = [j.descriptions.get(k) or k for k in j.keys]
                questions[name] = {"type": "score", "instructions": instruction, "criteria": criteria}
        return questions

    def _payload(self, request: Request) -> dict[str, Any]:
        if request.tools:
            raise self._refuse("tools", "tools have no slot on the systemone wire")
        cfg = request.config
        if cfg.tool_choice is not None:
            raise self._refuse("config.tool_choice", "tool_choice has no slot on the systemone wire")
        for name in _DROPPED_KNOBS:
            value = getattr(cfg, name)
            if value is None or value == ():
                continue
            adapt(f"config.{name}", "dropped", "no such control on the systemone wire (Jev returns decisions, not samples)",
                  asked=value if isinstance(value, (int, float, str, bool)) else str(value), provider=self.provider)
        questions = self._questions(request)
        payload: dict[str, Any] = {"model": request.model, "state": self._state(request), "questions": questions}
        if cfg.extensions:
            for key, value in cfg.extensions.items():
                if key == "n" and isinstance(value, (int, float)) and value > 1:
                    raise self._refuse("config.extensions.n", "n > 1 has no canonical multiple-response representation")
                payload[key] = value
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        request = self._wire_request(request)
        if stream:
            raise self._refuse("stream", "systemone answers in one piece; there is no stream to wrap")
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/v1/systemone",
            endpoint="systemone",
            model=request.model,
            headers={"Content-Type": "application/json"},
            payload=self._payload(request),
        )

    # ─── Response parsing (pure) ────────────────────────────────────

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        request = self._wire_request(request)
        data = response.json()
        found = request_judgments(request)

        def invalid(path: str, detail: str) -> ProviderError:
            error = ProviderError(
                f"malformed systemone reply at {path}: {detail}",
                provider=self.provider, status=response.status,
                request_id=self._request_id(response),
            )
            _attach_error_metadata(error, response.headers)
            return error

        def probability(raw: Any, path: str) -> float:
            # Check bounds before converting: arbitrarily large JSON integers
            # must be provider faults too, not float-conversion OverflowError.
            if (isinstance(raw, bool) or not isinstance(raw, (int, float))
                    or not 0 <= raw <= 1 or not math.isfinite(raw)):
                raise invalid(path, "expected a finite number in [0, 1]")
            return float(raw)

        if not isinstance(data, dict):
            raise invalid("$", "expected an object")
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise invalid("answers", "expected an object containing every declared judgment")
        if set(answers) != set(found):
            raise invalid("answers", "keys must match the declared judgments exactly")
        value: dict[str, Any] = {}
        probabilities: dict[str, dict[str, float]] = {}
        for name, j in found.items():
            answer = answers[name]
            path = f"answers.{name}"
            if not isinstance(answer, dict):
                raise invalid(path, "expected an answer object")
            expected = {"boolean": "noul", "choice": "choice", "ordered": "score"}[j.kind]
            if answer.get("type") != expected:
                raise invalid(f"{path}.type", f"expected {expected!r}")
            if j.kind == "boolean":
                p = probability(answer.get("noul"), f"{path}.noul")
                value[name] = p >= 0.5
                probabilities[name] = {"true": p, "false": 1.0 - p}
                continue
            dist = answer.get("probabilities")
            if not isinstance(dist, dict) or set(dist) != set(j.keys):
                raise invalid(f"{path}.probabilities", "expected one probability for every declared key, and no other keys")
            # INV-052: validate measurements individually, NEVER their total.
            probs = {k: probability(dist[k], f"{path}.probabilities.{k}") for k in j.keys}
            probabilities[name] = probs
            if j.kind == "choice":
                pick = answer.get("choice")
                if not isinstance(pick, str) or pick not in j.keys:
                    raise invalid(f"{path}.choice", "expected a declared choice key")
                value[name] = pick
            else:
                value[name] = int(max(probs, key=probs.get))
        try:
            part = DataPart(value=value, probabilities=probabilities or None,
                            method="provider_classification" if probabilities else None)
        except (TypeError, ValueError, OverflowError) as exc:
            raise invalid("answers", str(exc)) from exc
        usage_raw = data.get("usage")
        if usage_raw is None:
            usage_raw = {}
        if not isinstance(usage_raw, dict):
            raise invalid("usage", "expected an object or null")
        try:
            usage = Usage(input_tokens=usage_raw.get("input_tokens"), output_tokens=usage_raw.get("output_tokens"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise invalid("usage", str(exc)) from exc
        model = data.get("model")
        if model is not None and (not isinstance(model, str) or not model):
            raise invalid("model", "expected a non-empty string")
        provider_data: dict[str, Any] = {"typesafe": {"answers": answers}}
        try:
            return Response(
                id=self._request_id(response),
                model=model if model is not None else request.model,
                message=Message(role="assistant", parts=(part,)),
                finish_reason="stop",
                usage=usage,
                provider_data=provider_data,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise invalid("$", str(exc)) from exc

    @staticmethod
    def _request_id(response: HttpResponse) -> str | None:
        for key, val in response.headers:
            if key.lower() == "x-typesafe-request-id" and val:
                return val
        return None

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        raise self._refuse("stream", "systemone has no stream")

    # ─── Errors ─────────────────────────────────────────────────────

    def normalize_error(self, status: int, body: str) -> ProviderError:
        message = body.strip()[:500] or f"HTTP {status}"
        code: str | None = None
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            payload = None
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            code = detail.get("error_type") if isinstance(detail.get("error_type"), str) else None
            if isinstance(detail.get("message"), str):
                message = detail["message"]
        elif isinstance(detail, list) and detail:
            # pydantic validation: [{type, loc, msg, input}]
            first = detail[0] if isinstance(detail[0], dict) else {}
            loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
            message = f"{loc}: {first.get('msg', 'validation error')}" if loc else str(first.get("msg", message))
        kwargs: dict[str, Any] = {"provider": self.provider, "provider_code": code, "status": status}
        if status == 401 or code == "authentication_error":
            return AuthError(message, env_keys=self.access.env_keys, **kwargs)
        if status == 429:
            return RateLimitError(message, **kwargs)
        if status == 400 and "unknown model" in message.lower():
            return UnsupportedModelError(message, **kwargs)
        if status in (400, 422):
            return InvalidRequestError(message, **kwargs)
        if status >= 500:
            return ServerError(message, **kwargs)
        return self._with_login_hint(map_http_error(status, message, provider=self.provider,
                                                     env_keys=self.access.env_keys, provider_code=code))

    # ─── Models ─────────────────────────────────────────────────────

    def _models_request(self) -> TransportRequest:
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/v1/models",
            headers={"Content-Type": "application/json"},
        )

    def _models_from_body(self, body: str):
        data = json.loads(body)
        entries = data.get("models") if isinstance(data, dict) else None
        return model_infos_from_entries(
            entries, provider=self.provider, api_family="typesafe_systemone", id_of=lambda e: e.get("name"),
        )


__all__ = ["TypeSafeLM"]
