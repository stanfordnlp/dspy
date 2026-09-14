"""Conversions at the edge of the Request/Response contract, with lm15's own code.

Engines that speak an OpenAI-shaped SDK (LiteLLM) serialize Requests and read
responses here; fine-tuning files and history displays reuse the same writers.
No credentials are resolved and no network I/O happens here.
"""

import json
from typing import Any

from dspy._vendor.lm15.providers.base import HttpResponse
from dspy._vendor.lm15.serde import part_to_dict
from dspy.lm15 import OpenAIChatLM, OpenAILM, Request, response_from_openai_chat


class _NoTransport:
    def stream(self, request):
        raise RuntimeError("The typed conversion boundary must not perform network I/O")


def plain(obj: Any):
    """Read SDK fields without invoking deferred Pydantic serializers."""
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {key: plain(item) for key, item in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(item) for item in obj]
    return obj


def request_kwargs(request: Request, model_type: str) -> dict:
    """Serialize a Request as the keyword arguments of an OpenAI-shaped SDK call."""
    if model_type not in {"chat", "responses", "text"}:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    # Private dialect mapping is confined here because lm15 does not expose a
    # public outbound-body converter.
    dialect = OpenAIChatLM(api_key="conversion-only", transport=_NoTransport())
    if model_type == "responses":
        dialect = OpenAILM(api_key="conversion-only", transport=_NoTransport())
    data = dialect._payload(request, stream=False)
    data.pop("model", None)
    data.pop("stream", None)
    if model_type == "text":
        # Text-completion endpoints take one prompt; the conversation is
        # flattened the way DSPy always did for them.
        data["prompt"] = text_prompt(data.pop("messages"))
    return data


def text_prompt(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        lines.append(content or "")
    return "\n\n".join([*lines, "BEGIN RESPONSE:"])


def snapshot_request(request: Request) -> Request:
    """Read local media once so the cache key describes the bytes actually sent."""
    import base64
    from dataclasses import replace

    def snapshot(part):
        path = getattr(part, "path", None)
        if path is not None:
            return replace(part, path=None, data=base64.b64encode(path.read_bytes()).decode("ascii"))
        if part.type == "tool_result":
            content = tuple(snapshot(item) for item in part.content)
            if any(left is not right for left, right in zip(content, part.content, strict=True)):
                return replace(part, content=content)
        return part

    messages = []
    for message in request.messages:
        parts = tuple(snapshot(part) for part in message.parts)
        messages.append(replace(message, parts=parts) if any(left is not right for left, right in zip(parts, message.parts, strict=True))
                        else message)
    system = request.system
    if isinstance(system, tuple):
        frozen = tuple(snapshot(part) for part in system)
        if any(left is not right for left, right in zip(frozen, system, strict=True)):
            system = frozen
    if system is request.system and all(left is right for left, right in zip(messages, request.messages, strict=True)):
        return request
    return replace(request, messages=tuple(messages), system=system)


def history_messages(request: Request) -> list[dict]:
    """A display snapshot; never read local files while recording history."""
    messages = []
    if request.system is not None:
        system = request.system
        messages.append({"role": "system", "content": system if isinstance(system, str)
                         else [part_to_dict(part) for part in system]})
    for message in request.messages:
        messages.append({"role": message.role, "content": message.text if message.text is not None
                         else [part_to_dict(part) for part in message.parts]})
    return messages


def response_value(response, model_type: str, request: Request):
    """Read an OpenAI-shaped SDK response (Chat, Responses or text) as a Response."""
    body = plain(response)
    if model_type == "responses":
        dialect = OpenAILM(api_key="conversion-only", transport=_NoTransport())
        return dialect.parse_response(request, HttpResponse(
            status=200, reason="OK", headers=[], body=json.dumps(body).encode(),
        ))
    if model_type == "text":
        body = dict(body)
        body["choices"] = [
            {**choice, "message": {"role": "assistant", "content": choice.get("text")}}
            for choice in body.get("choices", [])
        ]
    return response_from_openai_chat(body, model=request.model)
