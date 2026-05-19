"""Anthropic language model backend for normalized DSPy requests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import anyio

from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LMCapabilities
from dspy.clients.language_models.types import (
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

AnthropicRequester = Callable[[dict[str, Any], bool], Any]

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_THINKING_BUDGET = 1024


class AnthropicLM(BaseLM):
    """Call Anthropic Messages with DSPy's normalized LM types.

    Args:
        model: Anthropic model name. A leading `anthropic/` prefix is removed
            before the provider call.
        api_key: Anthropic API key. If omitted, `ANTHROPIC_API_KEY` is used.
        api_base: Anthropic API base URL.
        api_version: Anthropic API version header.
        requester: Optional callable for tests or custom transports. It
            receives `(payload, stream)` and returns a provider response.
        temperature: Default sampling temperature for this LM.
        max_tokens: Default output-token budget for this LM.
        cache: Whether DSPy request memoization is enabled by default.
        callbacks: Optional DSPy callbacks.
        **kwargs: Additional default LM config values.
    """

    model_type = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str = "https://api.anthropic.com/v1",
        base_url: str | None = None,
        api_version: str = "2023-06-01",
        requester: AnthropicRequester | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        if base_url is not None:
            if api_base != "https://api.anthropic.com/v1" and api_base != base_url:
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
        self.api_version = api_version
        self._requester = requester

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(
            function_calling=True,
            reasoning=True,
            response_schema=True,
            streaming=True,
            input_image=True,
            input_file=True,
            tool_results=True,
        )

    def forward(self, request: LMRequest) -> LMResponse:
        payload = self.to_anthropic_request(request, stream=False)
        response = self._call(payload, stream=False)
        return self.from_anthropic_response(response, request)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        payload = self.to_anthropic_request(request, stream=True)
        yield from self.stream_to_events(self._call(payload, stream=True), request)

    async def aforward(self, request: LMRequest) -> LMResponse:
        return await anyio.to_thread.run_sync(self.forward, request)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        async for event in _async_iter_stream(self.forward_stream(request)):
            yield event

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        if isinstance(error, urllib.error.HTTPError):
            body = error.read().decode("utf-8", errors="replace")
            return _anthropic_error(error.code, body, request.model)
        return error

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        if self.api_base != "https://api.anthropic.com/v1":
            state["api_base"] = self.api_base
        if self.api_version != "2023-06-01":
            state["api_version"] = self.api_version
        return state

    def to_anthropic_request(self, request: LMRequest, *, stream: bool = False) -> dict[str, Any]:
        system_parts = []
        messages = []
        for message in request.messages:
            if message.role == "system":
                system_parts.extend(message.parts)
            else:
                messages.append(_message_to_anthropic(message))

        thinking_budget = _thinking_budget(request)
        payload: dict[str, Any] = {
            "model": _strip_provider_prefix(request.model, "anthropic"),
            "messages": messages,
            "max_tokens": _max_tokens(request, thinking_budget),
        }
        if stream:
            payload["stream"] = True
        if system_parts:
            payload["system"] = _parts_to_text(system_parts)
        config = request.config
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.stop:
            payload["stop_sequences"] = list(config.stop)
        if request.tools:
            payload["tools"] = [_tool_to_anthropic(tool) for tool in request.tools]
        if config.tool_choice is not None:
            payload["tool_choice"] = _tool_choice_to_anthropic(config.tool_choice)
        if thinking_budget is not None:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        if config.response_format is not None:
            payload["output_config"] = _response_format_to_anthropic(config.response_format)
        if config.prompt_cache is not None and config.prompt_cache.enabled is not False:
            _mark_anthropic_prompt_cache(payload)
        payload.update(config.extensions)
        return payload

    def from_anthropic_response(self, response: Any, request: LMRequest) -> LMResponse:
        data = _as_dict(response)
        parts = []
        for block in data.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(LMTextPart(text=str(block.get("text") or "")))
                parts.extend(_anthropic_citations(block))
            elif block_type == "tool_use":
                parts.append(
                    LMToolCallPart(
                        id=str(block.get("id")) if block.get("id") else None,
                        name=str(block.get("name") or ""),
                        args=block.get("input") if isinstance(block.get("input"), dict) else {},
                    )
                )
            elif block_type == "thinking":
                parts.append(LMThinkingPart(text=str(block.get("thinking") or block.get("text") or "")))
            elif block_type == "redacted_thinking":
                parts.append(LMThinkingPart(text="[redacted]", redacted=True))
        if not parts:
            parts = [LMTextPart(text="")]
        output = LMOutput(
            parts=parts,
            finish_reason=_anthropic_finish_reason(data.get("stop_reason"), has_tool_call=any(isinstance(p, LMToolCallPart) for p in parts)),
            truncated=data.get("stop_reason") in {"max_tokens", "model_context_window_exceeded"},
            provider_output=response,
        )
        usage_payload = data.get("usage", {}) or {}
        input_tokens = _int_or_none(usage_payload.get("input_tokens"))
        output_tokens = _int_or_none(usage_payload.get("output_tokens"))
        return LMResponse(
            model=data.get("model") or request.model,
            outputs=[output],
            usage=LMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens + output_tokens) if input_tokens is not None and output_tokens is not None else None,
                cache_read_tokens=_int_or_none(usage_payload.get("cache_read_input_tokens")),
                cache_write_tokens=_int_or_none(usage_payload.get("cache_creation_input_tokens")),
            ),
            response_id=data.get("id"),
            provider_response=response,
            provider_data=data,
        )

    def stream_to_events(self, stream: Any, request: LMRequest) -> Iterator[LMStreamEvent]:
        yielded_start = False
        usage = None
        cost = None
        for event in _iter_sse_payloads(stream):
            event_type = event.get("type")
            if event_type != "message_start" and not yielded_start:
                yielded_start = True
                yield LMStreamStartEvent(model=request.model)
            if event_type == "message_start":
                if not yielded_start:
                    message = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
                    yielded_start = True
                    yield LMStreamStartEvent(model=message.get("model") or request.model)
            elif event_type == "content_block_start":
                block = event.get("content_block", {}) if isinstance(event.get("content_block"), dict) else {}
                if block.get("type") == "tool_use":
                    yield LMStreamDeltaEvent(
                        output_index=0,
                        part_index=int(event.get("index", 0) or 0),
                        delta=LMToolCallDelta(
                            id=str(block.get("id")) if block.get("id") else None,
                            name=str(block.get("name")) if block.get("name") else None,
                            args_delta=json.dumps(block.get("input") or {}),
                        ),
                    )
            elif event_type == "content_block_delta":
                delta = event.get("delta", {}) if isinstance(event.get("delta"), dict) else {}
                part_index = int(event.get("index", 0) or 0)
                if delta.get("type") == "text_delta":
                    yield LMStreamDeltaEvent(output_index=0, part_index=part_index, delta=LMTextDelta(text=str(delta.get("text") or "")))
                elif delta.get("type") == "thinking_delta":
                    yield LMStreamDeltaEvent(output_index=0, part_index=part_index, delta=LMThinkingDelta(text=str(delta.get("thinking") or "")))
                elif delta.get("type") == "input_json_delta":
                    yield LMStreamDeltaEvent(output_index=0, part_index=part_index, delta=LMToolCallDelta(args_delta=str(delta.get("partial_json") or "")))
            elif event_type == "message_delta":
                delta = event.get("delta", {}) if isinstance(event.get("delta"), dict) else {}
                usage = _usage_from_anthropic(event.get("usage")) or usage
                yield LMStreamOutputEndEvent(
                    output_index=0,
                    finish_reason=_anthropic_finish_reason(delta.get("stop_reason")),
                    truncated=delta.get("stop_reason") in {"max_tokens", "model_context_window_exceeded"},
                )
            elif event_type == "message_stop":
                break
            elif event_type == "error":
                error = event.get("error", {}) if isinstance(event.get("error"), dict) else {}
                raise RuntimeError(str(error.get("message") or error or event))
        if not yielded_start:
            yield LMStreamStartEvent(model=request.model)
        yield LMStreamEndEvent(usage=usage, cost=cost)

    def _call(self, payload: dict[str, Any], *, stream: bool) -> Any:
        if self._requester is not None:
            return self._requester(payload, stream)
        headers = {
            "x-api-key": self.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }
        request = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=120 if stream else 60)
        if stream:
            return response
        return json.loads(response.read().decode("utf-8"))


def _message_to_anthropic(message: Any) -> dict[str, Any]:
    role = "assistant" if message.role == "assistant" else "user"
    if message.role == "developer":
        return {"role": "user", "content": [{"type": "text", "text": f"[developer]\n{_parts_to_text(message.parts)}"}]}
    return {"role": role, "content": [_part_to_anthropic(part) for part in message.parts]}


def _part_to_anthropic(part: Any) -> dict[str, Any]:
    if isinstance(part, LMTextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, LMImagePart):
        return {"type": "image", "source": _anthropic_source(part)}
    if isinstance(part, LMFilePart):
        return {"type": "document", "source": _anthropic_source(part)}
    if isinstance(part, LMToolCallPart):
        return {"type": "tool_use", "id": part.id or "", "name": part.name, "input": part.args}
    if isinstance(part, LMToolResultPart):
        content = [_tool_result_part_to_anthropic(item) for item in part.content]
        out: dict[str, Any] = {"type": "tool_result", "tool_use_id": part.call_id or ""}
        if len(content) == 1 and content[0].get("type") == "text":
            out["content"] = content[0]["text"]
        elif content:
            out["content"] = content
        if part.is_error:
            out["is_error"] = True
        return out
    if isinstance(part, LMThinkingPart):
        return {"type": "text", "text": part.text}
    return {"type": "text", "text": getattr(part, "text", "") or ""}


def _tool_result_part_to_anthropic(part: Any) -> dict[str, Any]:
    if isinstance(part, LMImagePart):
        return {"type": "image", "source": _anthropic_source(part)}
    if isinstance(part, LMFilePart):
        return {"type": "document", "source": _anthropic_source(part)}
    return {"type": "text", "text": getattr(part, "text", str(part))}


def _anthropic_source(part: LMImagePart | LMFilePart) -> dict[str, Any]:
    if part.url is not None:
        return {"type": "url", "url": part.url}
    if part.file_id is not None:
        return {"type": "file", "file_id": part.file_id}
    if part.data is not None:
        return {"type": "base64", "media_type": part.media_type, "data": part.data}
    if part.path is not None:
        with open(part.path, "rb") as file:
            import base64

            return {"type": "base64", "media_type": part.media_type, "data": base64.b64encode(file.read()).decode("ascii")}
    raise ValueError(f"{type(part).__name__} has no source.")


def _tool_to_anthropic(tool: Any) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}


def _tool_choice_to_anthropic(choice: Any) -> dict[str, Any]:
    if choice.mode == "none":
        return {"type": "none"}
    if choice.allowed and len(choice.allowed) == 1:
        return {"type": "tool", "name": choice.allowed[0]}
    if choice.mode == "required":
        return {"type": "any"}
    return {"type": "auto"}


def _thinking_budget(request: LMRequest) -> int | None:
    reasoning = request.config.reasoning
    if reasoning is None:
        return None
    return reasoning.max_tokens or _DEFAULT_THINKING_BUDGET


def _max_tokens(request: LMRequest, thinking_budget: int | None) -> int:
    visible = request.config.max_tokens or _DEFAULT_MAX_TOKENS
    return visible if thinking_budget is None else visible + thinking_budget


def _response_format_to_anthropic(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_json_schema"):
        return {"format": {"type": "json_schema", "schema": value.model_json_schema()}}
    return {"format": value}


def _mark_anthropic_prompt_cache(payload: dict[str, Any]) -> None:
    if payload.get("system") and isinstance(payload["system"], str):
        payload["system"] = [{"type": "text", "text": payload["system"], "cache_control": {"type": "ephemeral"}}]
        return
    messages = payload.get("messages") or []
    if messages and messages[0].get("content"):
        messages[0]["content"][-1].setdefault("cache_control", {"type": "ephemeral"})


def _anthropic_finish_reason(reason: Any, *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    if reason in {"max_tokens", "model_context_window_exceeded"}:
        return "length"
    if reason in {"tool_use", "pause_turn"}:
        return "tool_call"
    if reason in {"refusal", "safety", "content_filter"}:
        return "content_filter"
    return "stop"


def _anthropic_citations(block: dict[str, Any]) -> list[LMCitationPart]:
    citations = []
    for citation in block.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        text = citation.get("cited_text") or citation.get("text") or citation.get("quote")
        title = citation.get("title") or citation.get("document_title")
        url = citation.get("url") or citation.get("uri")
        if text or title or url:
            citations.append(LMCitationPart(text=text, title=title, url=url))
    return citations


def _usage_from_anthropic(value: Any) -> LMUsage | None:
    if not isinstance(value, dict):
        return None
    input_tokens = _int_or_none(value.get("input_tokens"))
    output_tokens = _int_or_none(value.get("output_tokens"))
    return LMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(input_tokens + output_tokens) if input_tokens is not None and output_tokens is not None else None,
    )


def _iter_sse_payloads(stream: Any) -> Iterator[dict[str, Any]]:
    if isinstance(stream, (list, tuple)):
        for item in stream:
            yield _as_dict(item)
        return
    event_lines: list[str] = []
    for raw_line in stream:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\n")
        if not line:
            for payload in _parse_sse_event(event_lines):
                yield payload
            event_lines = []
            continue
        event_lines.append(line)
    for payload in _parse_sse_event(event_lines):
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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _parts_to_text(parts: list[Any]) -> str:
    return "\n".join(getattr(part, "text", str(part)) for part in parts if getattr(part, "text", None) is not None)


def _strip_provider_prefix(model: str, provider: str) -> str:
    return model.removeprefix(f"{provider}/")


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _anthropic_error(status: int, body: str, model: str) -> Exception:
    from dspy.utils.exceptions import ContextWindowExceededError, LMAuthError, LMProviderError, LMRateLimitError

    try:
        data = json.loads(body)
        error = data.get("error", {}) if isinstance(data, dict) else {}
        message = str(error.get("message") or body)
    except Exception:
        message = body
    lowered = message.lower()
    if "context" in lowered or "too many tokens" in lowered or "prompt is too long" in lowered:
        return ContextWindowExceededError(model=model, provider="anthropic", message=message, status=status)
    if status == 401:
        return LMAuthError(model=model, provider="anthropic", message=message, status=status)
    if status == 429:
        return LMRateLimitError(model=model, provider="anthropic", message=message, status=status)
    return LMProviderError(model=model, provider="anthropic", message=message, status=status)
