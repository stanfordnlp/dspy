"""LiteLLM behind the same single-response interface as native lm15 engines."""

import inspect
import json
from dataclasses import replace

from dspy._vendor.lm15.sse import SSEEvent
from dspy.clients._litellm import get_litellm
from dspy.clients.engines.base import validate_request
from dspy.clients.engines.errors import wrap_error
from dspy.clients.legacy_outputs import plain
from dspy.clients.lm15_boundary import request_kwargs, response_value
from dspy.lm15 import OpenAIChatLM, Request, StreamEndEvent, StreamStartEvent
from dspy.utils.exceptions import LMError, LMProviderError, LMUnsupportedFeatureError


class _ChatStream:
    """Map chunks with the bundled dialect while keeping part indices distinct."""

    def __init__(self, request):
        self.request = request
        self.dialect = OpenAIChatLM(api_key="conversion-only", transport=None)
        self.indices = {}
        self.started = False
        self.terminal = False
        self.chunks = []

    def feed(self, chunk):
        body = plain(chunk)
        choices = body.get("choices") or []
        if len(choices) > 1 or any(choice.get("index", 0) != 0 for choice in choices):
            raise LMProviderError("A single-response engine received multiple streaming candidates.")
        for choice in choices:
            delta = choice.get("delta") or {}
            supported = {"role", "content", "reasoning_content", "reasoning", "tool_calls", "provider_specific_fields"}
            unknown = [key for key, val in delta.items() if key not in supported and val not in (None, "", [], {})]
            extras = delta.get("provider_specific_fields") or {}
            unknown.extend(f"provider_specific_fields.{key}" for key, val in extras.items()
                           if key != "citation" and val not in (None, "", [], {}))
            if unknown:
                raise LMUnsupportedFeatureError(
                    f"LiteLLM streaming fields have no implemented lm15 event mapping: {sorted(unknown)}",
                    model=self.request.model,
                )
        self.chunks.append(chunk)
        if not self.started:
            self.started = True
            yield StreamStartEvent(id=body.get("id") or None, model=body.get("model") or self.request.model)
        if choices and choices[0].get("finish_reason") is not None:
            self.terminal = True
        for event in self.dialect.parse_stream_events(
            self.request, SSEEvent(event=None, data=json.dumps(body)),
        ):
            if event.type == "error":
                from dspy._vendor.lm15.errors import error_class_for_code

                raise error_class_for_code(event.error.code)(event.error.message, provider_code=event.error.provider_code)
            if event.type != "delta":
                continue
            delta = event.delta
            key = (delta.type, delta.part_index)
            index = self.indices.setdefault(key, len(self.indices))
            yield replace(event, delta=replace(delta, part_index=index))
        # LiteLLM exposes Anthropic citations as a provider-specific delta.
        for choice in choices:
            delta = choice.get("delta") or {}
            citation = (delta.get("provider_specific_fields") or {}).get("citation")
            if isinstance(citation, dict):
                from dspy.lm15 import CitationDelta, StreamDeltaEvent

                text = citation.get("cited_text") or citation.get("text") or citation.get("supported_text")
                title = citation.get("document_title") or citation.get("title")
                url = citation.get("url")
                if not any((text, title, url)):
                    raise LMUnsupportedFeatureError("This citation chunk has no lm15 text, title or URL representation.")
                index = self.indices.setdefault(("citation", len(self.chunks)), len(self.indices))
                yield StreamDeltaEvent(CitationDelta(text=text, title=title, url=url, part_index=index))

    def finish(self, litellm):
        if not self.terminal:
            raise LMProviderError("LiteLLM stream ended without a completion reason; no successful end event emitted.")
        raw = litellm.stream_chunk_builder(self.chunks)
        response = response_value(raw, "chat", self.request)
        return StreamEndEvent(
            finish_reason=response.finish_reason, usage=response.usage, provider_data=response.provider_data,
        )


class _LiteLLMConfig:
    def __init__(self, *, model_type="chat", **client_options):
        if model_type not in {"chat", "responses"}:
            raise LMUnsupportedFeatureError(
                "Typed LiteLLM engines support chat and Responses APIs. Use ordinary LM calls for text completions."
            )
        allowed = {
            "api_key", "api_base", "base_url", "api_version", "organization", "project",
            "headers", "extra_headers", "extra_query", "timeout", "azure_ad_token_provider",
            "custom_llm_provider",
        }
        unknown = set(client_options) - allowed
        if unknown:
            raise TypeError(f"Unknown engine client options: {sorted(unknown)}. Generation options belong in Request.config.")
        self.model_type = model_type
        self.client_options = dict(client_options)
        self._closed = False

    def _arguments(self, request, *, streaming=False):
        validate_request(request)
        if self._closed:
            raise RuntimeError("Engine is closed")
        if streaming and self.model_type != "chat":
            raise LMUnsupportedFeatureError(
                "LiteLLMEngine streaming currently requires model_type='chat'. "
                "Use the native Responses engine for Responses streaming.", model=request.model,
            )
        data = request_kwargs(request, self.model_type)
        data.update(self.client_options)
        data.update(model=request.model, num_retries=0, cache={"no-cache": True, "no-store": True})
        if self.model_type == "chat":
            data["n"] = 1
        if streaming:
            data["stream"] = True
            data["stream_options"] = {"include_usage": True}
        return data

    def __repr__(self):
        return f"{type(self).__name__}(model_type={self.model_type!r})"


class LiteLLMEngine(_LiteLLMConfig):
    """One synchronous attempt. LiteLLM's process-global clients are borrowed."""

    def complete(self, request: Request):
        data = self._arguments(request)
        try:
            litellm = get_litellm(feature="LiteLLM engine")
            fn = litellm.responses if self.model_type == "responses" else litellm.completion
            return response_value(fn(**data), self.model_type, request)
        except LMError:
            raise
        except Exception as exc:
            raise wrap_error(exc, model=request.model) from exc

    def stream(self, request: Request):
        data = self._arguments(request, streaming=True)
        source = None
        try:
            litellm = get_litellm(feature="LiteLLM engine streaming")
            source = litellm.completion(**data)
            codec = _ChatStream(request)
            for chunk in source:
                yield from codec.feed(chunk)
            yield codec.finish(litellm)
        except LMError:
            raise
        except Exception as exc:
            raise wrap_error(exc, model=request.model) from exc
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    def close(self):
        self._closed = True


class AsyncLiteLLMEngine(_LiteLLMConfig):
    """Async equivalent; cancellation propagates without retrying."""

    async def complete(self, request: Request):
        data = self._arguments(request)
        try:
            litellm = get_litellm(feature="LiteLLM engine")
            fn = litellm.aresponses if self.model_type == "responses" else litellm.acompletion
            return response_value(await fn(**data), self.model_type, request)
        except LMError:
            raise
        except Exception as exc:
            raise wrap_error(exc, model=request.model) from exc

    async def stream(self, request: Request):
        data = self._arguments(request, streaming=True)
        source = None
        try:
            litellm = get_litellm(feature="LiteLLM engine streaming")
            source = await litellm.acompletion(**data)
            codec = _ChatStream(request)
            async for chunk in source:
                for event in codec.feed(chunk):
                    yield event
            yield codec.finish(litellm)
        except LMError:
            raise
        except Exception as exc:
            raise wrap_error(exc, model=request.model) from exc
        finally:
            close = getattr(source, "aclose", None) or getattr(source, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def aclose(self):
        self._closed = True
