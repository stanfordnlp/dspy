"""Conversions at the edge of the Request/Response contract, with lm15's own code.

OpenAI-shaped exports serialize Requests and read responses here;
fine-tuning files and history displays reuse the same writers.
No credentials are resolved and no network I/O happens here.
"""

import json
import mimetypes
from dataclasses import replace
from typing import Any

from dspy._vendor.lm15.providers.base import HttpResponse
from dspy._vendor.lm15.providers.common import media_base64, media_data_uri
from dspy._vendor.lm15.providers.openai_chat import _chat_content_parts
from dspy._vendor.lm15.serde import part_to_dict
from dspy.lm15 import (
    CitationPart,
    Message,
    OpenAIChatLM,
    OpenAILM,
    Request,
    TokenLogprob,
    TopLogprob,
    UnsupportedFeatureError,
    response_from_openai_chat,
)


class _NoTransport:
    def stream(self, request):
        raise RuntimeError("The typed conversion boundary must not perform network I/O")


class _SDKChatLM(OpenAIChatLM):
    """Preserve SDK/export media slots missing from the bundled Chat writer."""

    def _build_messages(self, request, compat, *, breakpoint_index=None):
        rows = []
        for index, message in enumerate(request.messages):
            media = message.role == "user" and any(part.type in {"audio", "document"} for part in message.parts)
            # Delegate every other message shape, including multi-row tool
            # results and system-prefix caching, to the bundled writer.
            rendered = super()._build_messages(
                replace(request, messages=(Message.user("") if media else message,),
                        system=request.system if index == 0 else None),
                compat, breakpoint_index=0 if index == breakpoint_index else None,
            )
            if media:
                content = []
                for part in message.parts:
                    if part.type == "audio":
                        if part.url is not None or part.file_id is not None:
                            raise UnsupportedFeatureError("Chat input_audio requires inline data or a local path")
                        format_ = part.media_type.split("/", 1)[-1].removeprefix("x-")
                        format_ = "mp3" if format_ == "mpeg" else format_
                        content.append({"type": "input_audio", "input_audio": {"data": media_base64(part), "format": format_}})
                    elif part.type == "document":
                        if part.url is not None:
                            raise UnsupportedFeatureError("Chat file inputs require file_id, inline data or a local path")
                        file = {"file_id": part.file_id} if part.file_id is not None else {
                            "file_data": media_data_uri(part),
                            "filename": "file" + (mimetypes.guess_extension(part.media_type) or ".bin"),
                        }
                        content.append({"type": "file", "file": file})
                    else:
                        content.extend(_chat_content_parts(replace(message, parts=(part,)), force_array=True))
                if index == breakpoint_index:
                    if not content or content[-1]["type"] != "text":
                        raise UnsupportedFeatureError("A Chat cache breakpoint requires a final text content block")
                    content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
                rendered[-1]["content"] = content
            rows.extend(rendered)
        return rows


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
    if model_type == "chat":
        dialect = _SDKChatLM(api_key="unused", transport=_NoTransport())
    if model_type == "responses":
        dialect = OpenAILM(api_key="conversion-only", transport=_NoTransport())
    data = dialect._payload(request, stream=False)
    data.pop("model", None)
    data.pop("stream", None)
    if model_type == "text":
        # Text-completion endpoints take one prompt; the conversation is
        # flattened the way DSPy always did for them.
        data["prompt"] = text_prompt(data.pop("messages"))
        if "max_completion_tokens" in data:
            data["max_tokens"] = data.pop("max_completion_tokens")
        if request.config.logprobs is not None:
            data["logprobs"] = request.config.logprobs
            data.pop("top_logprobs", None)
    return data


def text_prompt(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(block.get("type") != "text" for block in content):
                raise UnsupportedFeatureError("Text-completion prompts cannot carry non-text content blocks")
            content = "".join(block["text"] for block in content)
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


def repair_tool_arguments(response, raw_arguments=None):
    """Restore DSPy's JSON repair at the wire boundary, with raw provenance.

    ``partial_json`` is also a legitimate tool parameter. Only a malformed
    original argument string proves that lm15's input is a parser fallback.
    Streaming callers supply (id, name, raw) from their accumulator instead.
    """
    from dataclasses import replace

    import json_repair

    from dspy.lm15 import ToolCallPart

    if not response.tool_calls:
        return response
    if raw_arguments is None:
        raw_arguments = _raw_tool_arguments(response.provider_data or {})
    repaired = {}
    for call_id, name, raw in raw_arguments:
        if not isinstance(raw, str):
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            value = json_repair.loads(raw)
            if isinstance(value, dict):
                repaired[(call_id, name, raw)] = value
    parts = []
    changed = False
    for part in response.message.parts:
        if isinstance(part, ToolCallPart) and set(part.input) == {"partial_json"}:
            raw = part.input["partial_json"]
            value = repaired.get((part.id, part.name, raw)) if isinstance(raw, str) else None
            if value is not None:
                part = replace(part, input=value)
                changed = True
        parts.append(part)
    return replace(response, message=replace(response.message, parts=tuple(parts))) if changed else response


def _raw_tool_arguments(body):
    """Read only Chat/Responses function argument fields, correlated by call id."""
    for choice in body.get("choices", []) or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls", []) or []:
            if isinstance(call, dict) and call.get("type", "function") == "function":
                function = call.get("function")
                if isinstance(function, dict):
                    yield call.get("id"), function.get("name"), function.get("arguments")
    for item in body.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            yield item.get("call_id") or item.get("id"), item.get("name"), item.get("arguments")


def response_value(response, model_type: str, request: Request):
    """Read an OpenAI-shaped SDK response (Chat, Responses or text) as a Response."""
    body = plain(response)
    if model_type == "responses":
        dialect = OpenAILM(api_key="conversion-only", transport=_NoTransport())
        return repair_tool_arguments(dialect.parse_response(request, HttpResponse(
            status=200, reason="OK", headers=[], body=json.dumps(body).encode(),
        )))
    if model_type == "text":
        body = dict(body)
        body["choices"] = [
            {**choice, "message": {"role": "assistant", "content": choice.get("text")}}
            for choice in body.get("choices", [])
        ]
    parsed = response_from_openai_chat(body, model=request.model)
    if model_type == "text" and body.get("choices"):
        scores = body["choices"][0].get("logprobs")
        if scores is not None:
            tokens = scores.get("tokens") or []
            probabilities = scores.get("token_logprobs") or [None] * len(tokens)
            alternatives = scores.get("top_logprobs") or [None] * len(tokens)
            # Unknown scores (e.g. the first echoed token) cannot be invented.
            # The original arrays and offsets remain in provider_data.
            parsed = replace(parsed, logprobs=tuple(
                TokenLogprob(token=token, logprob=probability, top=tuple(
                    TopLogprob(token=alternative, logprob=score) for alternative, score in (top or {}).items()
                )) for token, probability, top in zip(tokens, probabilities, alternatives, strict=True)
                if probability is not None
            ), logprobs_complete=all(probability is not None for probability in probabilities))
    citations = []
    for choice in body.get("choices", []):
        fields = (choice.get("message") or {}).get("provider_specific_fields") or {}
        for group in fields.get("citations") or []:
            for citation in group if isinstance(group, list) else [group]:
                citations.append(CitationPart(
                    text=citation.get("cited_text") or citation.get("text") or citation.get("supported_text"),
                    title=citation.get("document_title") or citation.get("title"),
                    url=citation.get("url"),
                ))
    if citations:
        parsed = replace(parsed, message=replace(parsed.message, parts=(*parsed.message.parts, *citations)))
    return repair_tool_arguments(parsed)
