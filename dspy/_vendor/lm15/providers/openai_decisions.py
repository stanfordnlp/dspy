"""OpenAI Decisions: typed judgments, not text generation.

Wire contract: https://developers.openai.com/api/docs/guides/decisions
The Responses codec supplies conversation input; decisions have their own
questions, answers, capabilities, and endpoint. No SDK dependency is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from ..access import OPENAI_DECISIONS_API
from ..errors import ProviderError, UnsupportedFeatureError
from ..features import ProviderManifest
from ..judgments import non_judgment_properties, request_judgments
from ..types import Config, DataPart, ImagePart, Message, RefusalPart, Request, Response, TextPart
from .base import HttpResponse
from .common import parts_to_text, usage_from_responses
from .openai import OpenAILM


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


@dataclass
class OpenAIDecisionsLM(OpenAILM):
    provider: str = field(default="openai-decisions", init=False)
    manifest: ClassVar[ProviderManifest] = OPENAI_DECISIONS_API

    def __post_init__(self):
        self._compat_base = None
        super().__post_init__()

    def _refuse(self, feature: str, reason: str) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"OpenAI Decisions: {reason}", provider=self.provider, feature=feature)

    def _questions(self, request: Request) -> list[dict[str, Any]]:
        found = request_judgments(request)
        fmt = request.config.response_format or {}
        if not found or non_judgment_properties(fmt.get("schema"), found):
            raise self._refuse("response_format", "declare only boolean, choice, or ordered-score judgments")
        if len(found) > 400:
            raise self._refuse("questions", "at most 400 questions are supported")
        overrides = (request.config.extensions or {}).get("questions", {})
        if not isinstance(overrides, dict) or overrides.keys() - found.keys():
            raise self._refuse("questions", "question overrides must name declared judgments")
        questions = []
        for name, judgment in found.items():
            if judgment.kind == "box":
                raise self._refuse("response_format", "bounding boxes are not supported")
            kind = {"boolean": "predicate", "choice": "choice", "ordered": "score"}[judgment.kind]
            override = overrides.get(name, {})
            if not isinstance(override, dict) or override.keys() - {"type", "instructions", "criteria"}:
                raise self._refuse("questions", f"invalid question override for {name!r}")
            if override.get("type", kind) not in ({"noul", "predicate"} if kind == "predicate" else {kind}):
                raise self._refuse("questions", f"override changes the answer type for {name!r}")
            instruction = override.get("instructions", judgment.instruction or name)
            question = {"name": name, "type": kind, "instructions": _text(instruction)}
            criteria = override.get("criteria")
            if kind == "predicate":
                if criteria is not None:
                    question["instructions"] = _text({"instructions": instruction, "criteria": criteria})
            elif kind == "choice":
                if not 2 <= len(judgment.keys) <= 255:
                    raise self._refuse("questions", "choices require 2–255 options")
                criteria = criteria if criteria is not None else judgment.descriptions
                if not isinstance(criteria, dict) or set(criteria) != set(judgment.keys):
                    raise self._refuse("questions", f"choice criteria must preserve the keys of {name!r}")
                question["choices"] = [
                    {"value": key, **({"description": _text(criteria[key])} if criteria[key] is not None else {})}
                    for key in judgment.keys
                ]
            else:
                criteria = (
                    criteria if criteria is not None else [judgment.descriptions.get(k) or k for k in judgment.keys]
                )
                if (
                    not isinstance(criteria, list)
                    or len(criteria) != len(judgment.keys)
                    or not 2 <= len(criteria) <= 10
                ):
                    raise self._refuse("questions", "score criteria must preserve the declared 2–10 levels")
                question["levels"] = [
                    {"label": key, "description": _text(desc)} for key, desc in zip(judgment.keys, criteria)
                ]
            questions.append(question)
        return questions

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        if stream:
            raise self._refuse("stream", "streaming is not supported")
        if request.tools:
            raise self._refuse("tools", "decisions do not invoke tools")
        cfg = request.config
        defaults = Config()
        for item in fields(cfg):
            if item.name not in {"response_format", "extensions", "probabilities", "user_id"} and getattr(
                cfg, item.name
            ) != getattr(defaults, item.name):
                raise self._refuse(f"config.{item.name}", f"{item.name} is not supported")
        extensions = cfg.extensions or {}
        extra = extensions.keys() - {"questions", "prompt_cache_key", "prompt_cache_options", "safety_identifier"}
        if extra:
            raise self._refuse("config.extensions", f"unsupported options: {sorted(extra)}")
        # Admit only content this endpoint can read. Never let the broader
        # Responses codec silently drop or send unsupported media.
        for message in request.messages:
            for part in message.parts:
                allowed = (
                    (TextPart, DataPart, ImagePart)
                    if message.role == "user"
                    else (TextPart, RefusalPart)
                    if message.role == "assistant"
                    else (TextPart,)
                )
                if not isinstance(part, allowed) or message.role == "tool":
                    raise self._refuse("messages", "use text/JSON/images in user messages and text in other roles")
        inputs = self._build_input(request.messages, self._compat(request))
        if request.system:
            system = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            inputs.insert(0, {"role": "developer", "content": system})
        payload = {"model": request.model, "input": inputs, "questions": self._questions(request)}
        if cfg.user_id is not None:
            payload["safety_identifier"] = cfg.user_id
        payload.update({key: value for key, value in extensions.items() if key != "questions"})
        return payload

    def build_request(self, request: Request, stream: bool):
        request = self._wire_request(request)
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/decisions",
            endpoint="decisions",
            model=request.model,
            headers=self._headers(),
            payload=self._payload(request, stream),
        )

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        request = self._wire_request(request)
        data = response.json()
        questions = self._questions(request)

        def invalid(detail):
            return ProviderError(
                f"Malformed OpenAI Decisions reply: {detail}",
                provider=self.provider,
                status=response.status,
                request_id=response.header("x-request-id"),
            )

        def probability(value):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise invalid("probabilities must be finite numbers in [0, 1]")
            return float(value)

        if not isinstance(data, dict) or not isinstance(data.get("answers"), list):
            raise invalid("expected an answers array")
        raw_answers = data["answers"]
        if len(raw_answers) != len(questions):
            raise invalid("answer count differs from question count")
        values, distributions, evidence, refusals = {}, {}, {}, []
        for question, answer in zip(questions, raw_answers):
            name, kind = question["name"], question["type"]
            if not isinstance(answer, dict) or answer.get("name") != name:
                raise invalid("answers must match question names and order")
            if answer.get("type") == "refusal":
                refusals.append(name)
                continue
            if answer.get("type") != kind:
                raise invalid(f"wrong answer type for {name!r}")
            if kind == "predicate":
                p = probability(answer.get("probability"))
                values[name] = p >= 0.5
                distributions[name] = {"true": p, "false": 1 - p}
                evidence[name] = {"noul": p}
                continue
            raw = answer.get("probabilities")
            expected = (
                [x["value"] for x in question["choices"]] if kind == "choice" else list(range(len(question["levels"])))
            )
            if not isinstance(raw, list) or len(raw) != len(expected):
                raise invalid(f"incomplete distribution for {name!r}")
            probs = {}
            for entry in raw:
                if not isinstance(entry, dict):
                    raise invalid("expected probability objects")
                key = entry.get("value")
                if type(key) is not type(expected[0]) or key not in expected or str(key) in probs:
                    raise invalid(f"invalid or duplicate distribution key for {name!r}")
                if kind == "score" and entry.get("label") != question["levels"][key]["label"]:
                    raise invalid(f"invalid score label for {name!r}")
                probs[str(key)] = probability(entry.get("probability"))
            confidence = probability(answer.get("confidence"))
            if kind == "choice":
                value = answer.get("choice")
                if type(value) is not str or value not in expected:
                    raise invalid(f"invalid selected choice for {name!r}")
            else:
                score = answer.get("score")
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not 0 <= score <= len(expected) - 1
                ):
                    raise invalid(f"invalid score for {name!r}")
                value = int(max(probs, key=probs.get))
            values[name], distributions[name] = value, probs
            evidence[name] = {"probabilities": probs, "confidence": confidence}
        try:
            parts = [
                DataPart(
                    value=values,
                    probabilities=distributions or None,
                    method="provider_classification" if distributions else None,
                )
            ]
            if refusals:
                parts.append(RefusalPart(text="Decisions refused: " + ", ".join(refusals)))
            return Response(
                id=response.header("x-request-id"),
                model=data.get("model", request.model),
                message=Message(role="assistant", parts=tuple(parts)),
                finish_reason="content_filter" if refusals else "stop",
                usage=usage_from_responses(data.get("usage")),
                provider_data={self.provider: data, "decisions": {"answers": evidence, "refusals": refusals}},
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise invalid(str(error)) from error
