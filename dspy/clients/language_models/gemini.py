"""Google GenAI language model backend for normalized DSPy requests."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import anyio

from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LMCapabilities
from dspy.clients.language_models.types import (
    LMAudioPart,
    LMCitationPart,
    LMFilePart,
    LMImagePart,
    LMOutput,
    LMRequest,
    LMResponse,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMTextPart,
    LMThinkingDelta,
    LMThinkingPart,
    LMToolCallDelta,
    LMToolCallPart,
    LMToolResultPart,
    LMUsage,
)

GenAIRequester = Callable[[dict[str, Any], bool], Any]


class GenAILM(BaseLM):
    """Call Google GenAI with DSPy's normalized LM types.

    Args:
        model: Gemini model name. A leading `gemini/` or `google/` prefix is
            removed before the provider call.
        api_key: Google GenAI API key. If omitted, `GEMINI_API_KEY` or
            `GOOGLE_API_KEY` is used.
        api_base: Google GenAI API base URL.
        requester: Optional callable for tests or custom transports. It
            receives `(payload, stream)` and returns a provider response.
        temperature: Default sampling temperature for this LM.
        max_tokens: Default output-token budget for this LM.
        cache: Whether DSPy request memoization is enabled by default.
        callbacks: Optional DSPy callbacks.
        **kwargs: Additional default LM config values.
    """

    model_type = "genai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
        base_url: str | None = None,
        requester: GenAIRequester | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        if base_url is not None:
            if api_base != "https://generativelanguage.googleapis.com/v1beta" and api_base != base_url:
                raise ValueError("Pass only one of `api_base` or deprecated `base_url`, not both.")
            warnings.warn(
                "`base_url` is deprecated for normalized LMs; use `api_base` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            api_base = base_url
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )
        self.api_key = api_key
        self.api_base = api_base
        self._requester = requester

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(
            function_calling=True,
            reasoning=True,
            response_schema=True,
            streaming=True,
            input_image=True,
            input_audio=True,
            input_file=True,
            output_image=True,
            output_audio=True,
            tool_results=True,
        )

    def forward(self, request: LMRequest) -> LMResponse:
        payload = self.to_genai_request(request)
        response = self._call(payload, stream=False)
        return self.from_genai_response(response, request)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        payload = self.to_genai_request(request)
        yield from self.stream_to_events(self._call(payload, stream=True), request)

    async def aforward(self, request: LMRequest) -> LMResponse:
        return await anyio.to_thread.run_sync(self.forward, request)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        async for event in _async_iter_stream(self.forward_stream(request)):
            yield event

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        if isinstance(error, urllib.error.HTTPError):
            body = error.read().decode("utf-8", errors="replace")
            return _genai_error(error.code, body, request.model)
        return error

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        if self.api_base != "https://generativelanguage.googleapis.com/v1beta":
            state["api_base"] = self.api_base
        return state

    def to_genai_request(self, request: LMRequest) -> dict[str, Any]:
        system_parts = []
        contents = []
        for message in request.messages:
            if message.role == "system":
                system_parts.extend(message.parts)
            else:
                contents.append(_message_to_genai(message))

        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": _parts_to_text(system_parts)}]}

        generation_config = _generation_config(request)
        if generation_config:
            payload["generationConfig"] = generation_config
        if request.tools:
            payload["tools"] = [{"functionDeclarations": [_tool_to_genai(tool) for tool in request.tools]}]
        if request.config.tool_choice is not None:
            payload["toolConfig"] = _tool_choice_to_genai(request.config.tool_choice)
        if request.config.prompt_cache is not None and request.config.prompt_cache.enabled is not False:
            # Direct API cachedContents creation requires an extra request. Keep
            # the caller's preference visible as provider data instead of hiding
            # a second network call in request mapping.
            payload.setdefault("generationConfig", {}).setdefault("cachedContentPreference", {})["enabled"] = True
            if request.config.prompt_cache.key is not None:
                payload["generationConfig"]["cachedContentPreference"]["key"] = request.config.prompt_cache.key
        payload.update(request.config.extensions)
        return payload

    def from_genai_response(self, response: Any, request: LMRequest) -> LMResponse:
        data = _as_dict(response)
        candidate = _first_candidate(data)
        parts = _parse_genai_parts(candidate)
        full_text = "".join(part.text for part in parts if isinstance(part, LMTextPart))
        parts.extend(_genai_citations(candidate, full_text))
        if not parts:
            parts = [LMTextPart(text="")]
        output = LMOutput(
            parts=parts,
            finish_reason=_genai_finish_reason(candidate.get("finishReason"), has_tool_call=any(isinstance(part, LMToolCallPart) for part in parts)),
            truncated=candidate.get("finishReason") == "MAX_TOKENS",
            provider_output=candidate,
        )
        return LMResponse(
            model=request.model,
            outputs=[output],
            usage=_usage_from_genai(data.get("usageMetadata")),
            response_id=data.get("responseId"),
            provider_response=response,
            provider_data=data,
        )

    def stream_to_events(self, stream: Any, request: LMRequest) -> Iterator[LMStreamEvent]:
        yield LMStreamStartEvent(model=request.model)
        usage = None
        ended = False
        for payload in _iter_sse_payloads(stream):
            if "error" in payload:
                error = payload.get("error", {}) if isinstance(payload.get("error"), dict) else {}
                raise RuntimeError(str(error.get("message") or error or payload))
            candidate = _first_candidate(payload)
            content = candidate.get("content", {}) if isinstance(candidate.get("content"), dict) else {}
            for index, part in enumerate(content.get("parts", []) or []):
                if not isinstance(part, dict):
                    continue
                if part.get("thought") and "text" in part:
                    yield LMStreamDeltaEvent(output_index=0, part_index=index, delta=LMThinkingDelta(text=str(part.get("text") or "")))
                elif "text" in part:
                    yield LMStreamDeltaEvent(output_index=0, part_index=index, delta=LMTextDelta(text=str(part.get("text") or "")))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    call = part["functionCall"]
                    yield LMStreamDeltaEvent(
                        output_index=0,
                        part_index=index,
                        delta=LMToolCallDelta(
                            id=str(call.get("id")) if call.get("id") else None,
                            name=str(call.get("name")) if call.get("name") else None,
                            args_delta=json.dumps(call.get("args") or {}),
                        ),
                    )
            if candidate.get("finishReason"):
                ended = True
                usage = _usage_from_genai(payload.get("usageMetadata")) or usage
                yield LMStreamOutputEndEvent(
                    output_index=0,
                    finish_reason=_genai_finish_reason(candidate.get("finishReason")),
                    truncated=candidate.get("finishReason") == "MAX_TOKENS",
                )
            usage = _usage_from_genai(payload.get("usageMetadata")) or usage
        if not ended:
            yield LMStreamOutputEndEvent(output_index=0)
        yield LMStreamEndEvent(usage=usage)

    def _call(self, payload: dict[str, Any], *, stream: bool) -> Any:
        if self._requester is not None:
            return self._requester(payload, stream)
        endpoint = "streamGenerateContent" if stream else "generateContent"
        params = {"key": self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")}
        if stream:
            params["alt"] = "sse"
        url = f"{self.api_base.rstrip('/')}/{_model_path(self.model)}:{endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=120 if stream else 60)
        if stream:
            return response
        return json.loads(response.read().decode("utf-8"))


def _message_to_genai(message: Any) -> dict[str, Any]:
    if message.role == "developer":
        return {"role": "user", "parts": [{"text": f"[developer]\n{_parts_to_text(message.parts)}"}]}
    role = "model" if message.role == "assistant" else "user"
    return {"role": role, "parts": [_part_to_genai(part) for part in message.parts]}


def _part_to_genai(part: Any) -> dict[str, Any]:
    if isinstance(part, LMTextPart):
        return {"text": part.text}
    if isinstance(part, LMImagePart | LMAudioPart | LMFilePart):
        mime_type = part.media_type or "application/octet-stream"
        if part.url is not None:
            return {"fileData": {"mimeType": mime_type, "fileUri": part.url}}
        if part.file_id is not None:
            return {"fileData": {"mimeType": mime_type, "fileUri": part.file_id}}
        if part.data is not None:
            return {"inlineData": {"mimeType": mime_type, "data": part.data}}
        if part.path is not None:
            with open(part.path, "rb") as file:
                return {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(file.read()).decode("ascii")}}
    if isinstance(part, LMToolCallPart):
        payload = {"name": part.name, "args": part.args}
        if part.id is not None:
            payload["id"] = part.id
        return {"functionCall": payload}
    if isinstance(part, LMToolResultPart):
        response = {"name": part.name or "tool", "response": {"result": _parts_to_text(part.content)}}
        if part.call_id is not None:
            response["id"] = part.call_id
        return {"functionResponse": response}
    if isinstance(part, LMThinkingPart):
        return {"text": part.text, "thought": True}
    return {"text": getattr(part, "text", "") or ""}


def _generation_config(request: LMRequest) -> dict[str, Any]:
    config = request.config
    generation_config: dict[str, Any] = {}
    if config.temperature is not None:
        generation_config["temperature"] = config.temperature
    if config.max_tokens is not None:
        generation_config["maxOutputTokens"] = config.max_tokens
    if config.top_p is not None:
        generation_config["topP"] = config.top_p
    if config.stop:
        generation_config["stopSequences"] = list(config.stop)
    if config.response_format is not None:
        generation_config.update(_response_format_to_genai(config.response_format))
    if config.reasoning is not None:
        thinking: dict[str, Any] = {"includeThoughts": True}
        if config.reasoning.max_tokens is not None:
            thinking["thinkingBudget"] = config.reasoning.max_tokens
        generation_config["thinkingConfig"] = thinking
    return generation_config


def _response_format_to_genai(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if value.get("type") == "json_object":
            return {"responseMimeType": "application/json"}
        if value.get("type") == "json_schema" and isinstance(value.get("schema"), dict):
            return {"responseMimeType": "application/json", "responseJsonSchema": value["schema"]}
        return dict(value)
    if hasattr(value, "model_json_schema"):
        return {"responseMimeType": "application/json", "responseJsonSchema": value.model_json_schema()}
    return {"responseMimeType": str(value)}


def _tool_to_genai(tool: Any) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "parameters": tool.parameters}


def _tool_choice_to_genai(choice: Any) -> dict[str, Any]:
    mode = {"none": "NONE", "required": "ANY", "auto": "AUTO"}[choice.mode]
    config: dict[str, Any] = {"mode": mode}
    if choice.allowed:
        config["allowedFunctionNames"] = list(choice.allowed)
    return {"functionCallingConfig": config}


def _parse_genai_parts(candidate: dict[str, Any]) -> list[Any]:
    content = candidate.get("content", {}) if isinstance(candidate.get("content"), dict) else {}
    parts = []
    for part in content.get("parts", []) or []:
        if not isinstance(part, dict):
            continue
        if part.get("thought") and "text" in part:
            parts.append(LMThinkingPart(text=str(part.get("text") or "")))
        elif "text" in part:
            parts.append(LMTextPart(text=str(part.get("text") or "")))
        elif "functionCall" in part and isinstance(part["functionCall"], dict):
            call = part["functionCall"]
            parts.append(
                LMToolCallPart(
                    id=str(call.get("id")) if call.get("id") else None,
                    name=str(call.get("name") or ""),
                    args=call.get("args") if isinstance(call.get("args"), dict) else {},
                )
            )
        elif "inlineData" in part and isinstance(part["inlineData"], dict):
            inline = part["inlineData"]
            mime_type = str(inline.get("mimeType") or "application/octet-stream")
            data = str(inline.get("data") or "")
            if mime_type.startswith("image/"):
                parts.append(LMImagePart(data=data, media_type=mime_type))
            elif mime_type.startswith("audio/"):
                parts.append(LMAudioPart(data=data, media_type=mime_type))
            else:
                parts.append(LMFilePart(data=data, media_type=mime_type))
        elif "fileData" in part and isinstance(part["fileData"], dict):
            file_data = part["fileData"]
            mime_type = str(file_data.get("mimeType") or "application/octet-stream")
            uri = str(file_data.get("fileUri") or "")
            if mime_type.startswith("image/"):
                parts.append(LMImagePart(url=uri, media_type=mime_type))
            elif mime_type.startswith("audio/"):
                parts.append(LMAudioPart(url=uri, media_type=mime_type))
            else:
                parts.append(LMFilePart(url=uri, media_type=mime_type))
    return parts


def _genai_citations(candidate: dict[str, Any], full_text: str) -> list[LMCitationPart]:
    grounding = candidate.get("groundingMetadata")
    if not isinstance(grounding, dict):
        return []
    chunks = grounding.get("groundingChunks") or []
    supports = grounding.get("groundingSupports") or []
    if not isinstance(chunks, list) or not isinstance(supports, list):
        return []
    citations = []
    for support in supports:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment") if isinstance(support.get("segment"), dict) else {}
        text = _segment_text(segment, full_text)
        for index in support.get("groundingChunkIndices") or []:
            idx = _int_or_none(index)
            if idx is None or idx < 0 or idx >= len(chunks) or not isinstance(chunks[idx], dict):
                continue
            source = chunks[idx].get("web") or chunks[idx].get("retrievedContext") or {}
            if isinstance(source, dict) and (source.get("uri") or source.get("title") or text):
                citations.append(LMCitationPart(text=text, title=source.get("title"), url=source.get("uri") or source.get("url")))
    return citations


def _segment_text(segment: dict[str, Any], full_text: str) -> str | None:
    if isinstance(segment.get("text"), str) and segment["text"]:
        return segment["text"]
    start = _int_or_none(segment.get("startIndex"))
    end = _int_or_none(segment.get("endIndex"))
    if start is not None and end is not None and 0 <= start < end <= len(full_text):
        return full_text[start:end]
    return None


def _usage_from_genai(value: Any) -> LMUsage | None:
    if not isinstance(value, dict):
        return None
    return LMUsage(
        input_tokens=_int_or_none(value.get("promptTokenCount")),
        output_tokens=_int_or_none(value.get("candidatesTokenCount") or value.get("responseTokenCount")),
        total_tokens=_int_or_none(value.get("totalTokenCount")),
        cache_read_tokens=_int_or_none(value.get("cachedContentTokenCount")),
        reasoning_tokens=_int_or_none(value.get("thoughtsTokenCount")),
    )


def _genai_finish_reason(reason: Any, *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    reason = str(reason or "").upper()
    if reason == "MAX_TOKENS":
        return "length"
    if reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        return "content_filter"
    return "stop"


def _iter_sse_payloads(stream: Any) -> Iterator[dict[str, Any]]:
    if isinstance(stream, (list, tuple)):
        for item in stream:
            yield _as_dict(item)
        return
    lines: list[str] = []
    for raw_line in stream:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\n")
        if not line:
            for payload in _parse_sse_event(lines):
                yield payload
            lines = []
            continue
        lines.append(line)
    for payload in _parse_sse_event(lines):
        yield payload


async def _async_iter_stream(events: Iterator[LMStreamEvent]) -> AsyncIterator[LMStreamEvent]:
    sentinel = object()
    iterator = iter(events)
    while True:
        event = await anyio.to_thread.run_sync(_next_or_sentinel, iterator, sentinel)
        if event is sentinel:
            break
        yield event


def _next_or_sentinel(iterator: Iterator[Any], sentinel: object) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return sentinel


def _parse_sse_event(lines: list[str]) -> Iterator[dict[str, Any]]:
    data = "\n".join(line.removeprefix("data:").strip() for line in lines if line.startswith("data:"))
    if not data or data == "[DONE]":
        return
    yield json.loads(data)


def _first_candidate(data: dict[str, Any]) -> dict[str, Any]:
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    return candidate


def _model_path(model: str) -> str:
    model = model.removeprefix("gemini/").removeprefix("google/")
    return model if model.startswith("models/") else f"models/{model}"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _parts_to_text(parts: list[Any]) -> str:
    return "\n".join(getattr(part, "text", str(part)) for part in parts if getattr(part, "text", None) is not None)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _genai_error(status: int, body: str, model: str) -> Exception:
    from dspy.utils.exceptions import ContextWindowExceededError, LMAuthError, LMProviderError, LMRateLimitError

    try:
        data = json.loads(body)
        error = data.get("error", {}) if isinstance(data, dict) else {}
        message = str(error.get("message") or body)
    except Exception:
        message = body
    lowered = message.lower()
    if "context" in lowered or ("token" in lowered and ("limit" in lowered or "exceed" in lowered)):
        return ContextWindowExceededError(model=model, provider="gemini", message=message, status=status)
    if status in {401, 403}:
        return LMAuthError(model=model, provider="gemini", message=message, status=status)
    if status == 429:
        return LMRateLimitError(model=model, provider="gemini", message=message, status=status)
    return LMProviderError(model=model, provider="gemini", message=message, status=status)
