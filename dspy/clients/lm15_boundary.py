"""Conversions at the edge of the Request/Response contract, with lm15's own code.

Engines that speak an OpenAI-shaped SDK (LiteLLM) serialize Requests and read
responses here; fine-tuning files and history displays reuse the same writers.
No credentials are resolved and no network I/O happens here.
"""

import json
import mimetypes
from dataclasses import replace

from dspy._vendor.lm15.providers.base import HttpResponse
from dspy._vendor.lm15.providers.common import media_base64, media_data_uri
from dspy._vendor.lm15.providers.openai_chat import _chat_content_parts
from dspy._vendor.lm15.serde import part_to_dict
from dspy.clients.legacy_outputs import plain
from dspy.lm15 import (
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

    # TODO(upstream-lm15): Add native Chat audio/file serialization to lm15,
    # then remove this temporary subclass once the bundled version includes it.
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


def request_kwargs(request: Request, model_type: str) -> dict:
    """Serialize a Request as the keyword arguments of an OpenAI-shaped SDK call."""
    if "n" in (request.config.extensions or {}):
        raise ValueError("A typed Request produces one Response; do not put n in Config.extensions.")
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
    return parsed
