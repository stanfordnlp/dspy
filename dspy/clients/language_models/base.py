"""Normalized language model contract for DSPy."""

from __future__ import annotations

import copy
import datetime
import json
import logging
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

import anyio
from typing_extensions import Self

from dspy.dsp.utils import settings
from dspy.utils.callback import ACTIVE_CALL_ID
from dspy.utils.inspect_history import pretty_print_history

if TYPE_CHECKING:
    from dspy.clients.language_models.types import (
        AsyncLMStream,
        LMRequest,
        LMResponse,
        LMStream,
        LMStreamEvent,
    )
    from dspy.utils.callback import BaseCallback


MAX_HISTORY_SIZE = 10_000
GLOBAL_LANGUAGE_MODEL_HISTORY: list[Mapping[str, Any]] = []

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LMCapabilities:
    """Optional model and deployment metadata for an LM backend.

    Capabilities are descriptive hints. They can help adapters choose a native
    path, but the concrete `LanguageModel` still decides how to handle each
    request in `forward()`.

    Attributes:
        function_calling: Whether the model can return native tool calls.
        reasoning: Whether the model can return native reasoning content.
        response_schema: Whether the model can enforce a structured response
            schema, such as JSON schema or a Pydantic model.
        streaming: Whether the model can stream normalized DSPy events.
        input_image: Whether request messages may contain image parts.
        input_audio: Whether request messages may contain audio parts.
        input_file: Whether request messages may contain file parts.
        output_image: Whether responses may contain generated image parts.
        output_audio: Whether responses may contain generated audio parts.
        tool_results: Whether request messages may contain tool-result parts.
        extensions: Extra backend-specific metadata.
    """

    function_calling: bool = False
    reasoning: bool = False
    response_schema: bool = False
    streaming: bool = False
    input_image: bool = False
    input_audio: bool = False
    input_file: bool = False
    output_image: bool = False
    output_audio: bool = False
    tool_results: bool = False
    extensions: dict[str, Any] = field(default_factory=dict)


class LanguageModel:
    """Call a language model with a normalized DSPy request.

    Subclass `LanguageModel` when you want a model implementation to behave
    like a native DSPy LM. Users call the object with friendly inputs such as
    strings, `dspy.User(...)` messages, or an `LMRequest`. The base class
    normalizes those inputs, calls `forward(request)`, records history, and
    returns an `LMResponse`.

    Concrete subclasses implement `forward()`. They translate `LMRequest` into
    their provider or runtime format, run the model, and return `LMResponse`.

    Args:
        model: The model name or deployment identifier used by this LM.
        temperature: Default sampling temperature for requests made through
            this LM. Per-call `temperature=` overrides this value.
        max_tokens: Default output-token budget for requests made through this
            LM. Per-call `max_tokens=` overrides this value.
        cache: Whether this LM should use DSPy's cache by default.
        callbacks: Optional callbacks attached to this LM instance. Callback
            execution is wired by DSPy's callback layer when this class is
            exported as a public LM type.
        **kwargs: Additional default request configuration or provider-specific
            values.

    Examples:
        Define a small custom LM:
        ```python
        import dspy


        class EchoLM(dspy.LanguageModel):
            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                return dspy.LMResponse.from_text("hello", model=request.model)


        lm = EchoLM(model="test/echo")
        response = lm("Say hello")
        print(response.text)
        ```

    See Also:
        [`dspy.LM`][dspy.LM]
        [`dspy.LMRequest`][dspy.LMRequest]
        [`dspy.LMResponse`][dspy.LMResponse]
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        self.model = model
        self.cache = cache
        self.callbacks = callbacks or []
        self.num_retries = num_retries
        self.kwargs = _default_lm_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history: list[dict[str, Any]] = []
        self._warned_zero_temp_rollout = False

    # ---------------------------------------------------------------------
    # Model and deployment metadata
    # ---------------------------------------------------------------------

    @property
    def capabilities(self) -> LMCapabilities:
        """The native metadata available for this model instance."""
        return self.get_capabilities()

    def get_capabilities(self) -> LMCapabilities:
        """Return optional native model and deployment hints."""
        return LMCapabilities()

    # ---------------------------------------------------------------------
    # Core execution hooks for concrete LanguageModel implementations
    # ---------------------------------------------------------------------

    def forward(self, request: LMRequest) -> LMResponse:
        """Run one normalized language model request.

        Subclasses must implement this method. Translate `request` into the
        provider format, call the provider, and return an `LMResponse`.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement forward(request).")

    async def aforward(self, request: LMRequest) -> LMResponse:
        """Run one normalized language model request asynchronously.

        Subclasses that support native async inference should override this
        method. The default raises `NotImplementedError` rather than silently
        running sync inference in an event loop.
        """
        raise NotImplementedError("Subclasses must implement aforward(request) for async calls.")

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        """Run one normalized language model request as a stream of events."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming.")

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        """Run one normalized language model request as an async stream of events."""
        raise NotImplementedError(f"{type(self).__name__} does not support async streaming.")

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        """Map a provider exception to a DSPy exception.

        Subclasses should override this when their provider raises native
        exceptions for conditions DSPy understands, such as context-window
        failures. The default returns the original exception unchanged.
        """
        return error

    # ---------------------------------------------------------------------
    # State, serialization, and lifecycle hooks
    # ---------------------------------------------------------------------

    def dump_state(self) -> dict[str, Any]:
        """Return a sanitized reconstruction state for this LM.

        The default state works for simple constructor-only LMs. Subclasses with
        clients, local weights, auth objects, or other runtime resources should
        override this method and pair it with `load_state()`.
        """
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key != "api_key"}
        return {
            "model": self.model,
            "cache": self.cache,
            "num_retries": self.num_retries,
            **filtered_kwargs,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Self:
        """Reconstruct this LM from `dump_state()` output."""
        return cls(**state)

    def copy(self, **overrides: Any) -> Self:
        """Return a runtime copy of this LM with updated inference defaults.

        The copy keeps provider/runtime resources by reference, then isolates
        mutable DSPy-owned state such as history, callbacks, and default request
        kwargs. This makes `lm.copy(...)` safe for provider clients, sessions,
        local model handles, and other resources that cannot or should not be
        deep-copied.

        Subclasses with unusual resource ownership can override this method, but
        should preserve the same public semantics: no provider call, fresh
        history, independent kwargs, and the supplied overrides applied as LM
        attributes or request defaults.
        """
        new_instance = copy.copy(self)
        new_instance.history = []
        new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
        new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})

        for key, value in overrides.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if key in new_instance.kwargs or not hasattr(self, key):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value

        return new_instance

    # ---------------------------------------------------------------------
    # Public call API
    # ---------------------------------------------------------------------

    def __call__(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMResponse:
        """Normalize a direct LM call and return an `LMResponse`.

        Args:
            *items: Positional content such as text, `dspy.Image`, message
                constructors like `dspy.User(...)`, or an `LMRequest`.
            prompt: Optional text prompt keyword.
            messages: Optional chat messages accepted at the public boundary
                and normalized immediately.
            request: An explicit normalized request.
            **kwargs: Request configuration that overrides this LM's defaults.

        Returns:
            An `LMResponse` with normalized outputs, usage, cost, cache status,
            and provider metadata when available.
        """
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        call_id = self._start_lm_callbacks(
            callbacks,
            request=normalized_request,
            raw_inputs=self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs),
        )
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        result = None
        exception = None
        try:
            response = self._forward_with_retry(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def acall(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMResponse:
        """Normalize an async LM call and return an `LMResponse`."""
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        call_id = self._start_lm_callbacks(
            callbacks,
            request=normalized_request,
            raw_inputs=self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs),
        )
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        result = None
        exception = None
        try:
            response = await self._aforward_with_retry(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    def normalize_request(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMRequest:
        """Normalize public call inputs into an `LMRequest`.

        This method is deliberately small and delegates most parsing to
        `LMRequest`. The request type owns the content vocabulary; the language
        model owns defaults such as `model`, `cache`, and generation kwargs.
        """
        lm_types = _import_lm_types()
        request_cls = lm_types.LMRequest

        if request is None and items and isinstance(items[0], request_cls):
            request = items[0]
            items = items[1:]

        if request is not None:
            if prompt is not None or messages is not None or items:
                raise ValueError(
                    "Pass either an LMRequest or direct-call inputs, not both. "
                    "Use call kwargs to override request config."
                )
            normalized = self._override_request(request, **kwargs)
            self._warn_zero_temp_rollout(normalized)
            return normalized

        merged_kwargs = {**self.kwargs, **kwargs}
        merged_kwargs.setdefault("cache", self.cache)

        if hasattr(request_cls, "from_call"):
            normalized = request_cls.from_call(
                model=self.model,
                items=items,
                prompt=prompt,
                messages=messages,
                **merged_kwargs,
            )
            self._warn_zero_temp_rollout(normalized)
            return normalized

        if hasattr(request_cls, "from_prompt_or_messages"):
            if items:
                raise TypeError(
                    "Positional LM items require LMRequest.from_call(). "
                    "Implement that constructor in dspy.clients.language_models.types."
                )
            normalized = request_cls.from_prompt_or_messages(
                model=self.model,
                prompt=prompt,
                messages=messages,
                **merged_kwargs,
            )
            self._warn_zero_temp_rollout(normalized)
            return normalized

        raise TypeError("LMRequest must define from_call() or from_prompt_or_messages().")

    def stream(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMStream:
        """Normalize a direct LM call and return a stream with `.result()`."""
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        events = self._cached_stream_events(normalized_request, mode="stream")
        if events is None:
            try:
                self._require_stream_support(async_=False)
            except Exception as error:
                self._observe_failed_stream_construction(
                    normalized_request,
                    error,
                    callbacks=callbacks,
                    raw_inputs=raw_inputs,
                )
                raise
            events = self._cache_wrapped_stream_events(
                normalized_request,
                self.forward_stream(normalized_request),
                mode="stream",
            )
        lm_types = _import_lm_types()
        return lm_types.LMStream(
            request=normalized_request,
            events=self._callback_wrapped_stream_events(
                normalized_request,
                events,
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    def astream(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> AsyncLMStream:
        """Normalize a direct LM call and return an async stream with `.result()`.

        `astream()` returns an async iterator directly, so callers can write
        `async for event in lm.astream(...)` without first awaiting the stream.
        """
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        events = self._cached_astream_events(normalized_request, mode="astream")
        if events is None:
            try:
                self._require_stream_support(async_=True)
            except Exception as error:
                self._observe_failed_stream_construction(
                    normalized_request,
                    error,
                    callbacks=callbacks,
                    raw_inputs=raw_inputs,
                )
                raise
            events = self._cache_wrapped_astream_events(
                normalized_request,
                self.aforward_stream(normalized_request),
                mode="astream",
            )
        lm_types = _import_lm_types()
        return lm_types.AsyncLMStream(
            request=normalized_request,
            events=self._callback_wrapped_astream_events(
                normalized_request,
                events,
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    # ---------------------------------------------------------------------
    # Callback machinery
    # ---------------------------------------------------------------------

    def _get_active_callbacks(self) -> list[BaseCallback]:
        """Return global and instance callbacks for this LM call."""
        return list(settings.get("callbacks", []) or []) + list(getattr(self, "callbacks", []) or [])

    def _raw_callback_inputs(
        self,
        *,
        items: tuple[Any, ...],
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        raw: dict[str, Any] = {"items": items, "prompt": prompt, "messages": messages, "kwargs": kwargs}
        return _sanitize_callback_value(raw)

    def _start_lm_callbacks(
        self,
        callbacks: list[BaseCallback],
        *,
        request: LMRequest,
        raw_inputs: dict[str, Any],
    ) -> str | None:
        if not callbacks:
            return None

        call_id = uuid.uuid4().hex
        inputs = {"request": _sanitize_lm_request_for_callbacks(request), "raw": raw_inputs}
        for callback in callbacks:
            try:
                callback.on_lm_start(call_id=call_id, instance=self, inputs=inputs)
            except Exception as error:
                logger.warning("Error when calling callback %s: %s", callback, error)
        return call_id

    def _end_lm_callbacks(
        self,
        callbacks: list[BaseCallback],
        *,
        call_id: str | None,
        outputs: LMResponse | None,
        exception: Exception | None,
    ) -> None:
        if not callbacks or call_id is None:
            return

        for callback in callbacks:
            try:
                callback.on_lm_end(call_id=call_id, outputs=outputs, exception=exception)
            except Exception as error:
                logger.warning("Error when applying callback %s's LM end handler: %s", callback, error)

    def _observe_failed_stream_construction(
        self,
        request: LMRequest,
        error: Exception,
        *,
        callbacks: list[BaseCallback],
        raw_inputs: dict[str, Any],
    ) -> None:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        try:
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=None, exception=error)
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)

    def _callback_wrapped_stream_events(
        self,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        *,
        callbacks: list[BaseCallback],
        raw_inputs: dict[str, Any],
    ) -> Iterator[LMStreamEvent]:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        builder = _import_lm_types().LMOutputBuilder()
        result = None
        exception = None
        try:
            for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def _callback_wrapped_astream_events(
        self,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        *,
        callbacks: list[BaseCallback],
        raw_inputs: dict[str, Any],
    ) -> AsyncIterator[LMStreamEvent]:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        builder = _import_lm_types().LMOutputBuilder()
        result = None
        exception = None
        try:
            async for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    # ---------------------------------------------------------------------
    # History and internal execution machinery
    # ---------------------------------------------------------------------

    def inspect_history(self, n: int = 1, file: Any | None = None) -> None:
        """Print recent LM interactions recorded on this instance."""
        pretty_print_history(self.history, n, file=file)

    def update_history(self, entry: dict[str, Any]) -> None:
        """Append one normalized interaction to DSPy's LM history stores."""
        if settings.disable_history:
            return

        if len(GLOBAL_LANGUAGE_MODEL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_LANGUAGE_MODEL_HISTORY.pop(0)
        GLOBAL_LANGUAGE_MODEL_HISTORY.append(entry)

        if settings.max_history_size != 0:
            if len(self.history) >= settings.max_history_size:
                self.history.pop(0)
            self.history.append(entry)

        for module in settings.caller_modules or []:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)

    def _forward_with_retry(self, request: LMRequest) -> LMResponse:
        attempts = max(0, int(getattr(self, "num_retries", 0) or 0)) + 1
        for attempt in range(attempts):
            try:
                return self._forward_with_cache(request)
            except Exception as error:
                normalized_error = self.normalize_error(error, request)
                if attempt >= attempts - 1 or not _is_retryable_lm_error(normalized_error):
                    if normalized_error is error:
                        raise
                    raise normalized_error from error
                _sleep_before_retry(attempt)
        raise RuntimeError("unreachable")

    async def _aforward_with_retry(self, request: LMRequest) -> LMResponse:
        attempts = max(0, int(getattr(self, "num_retries", 0) or 0)) + 1
        for attempt in range(attempts):
            try:
                return await self._aforward_with_cache(request)
            except Exception as error:
                normalized_error = self.normalize_error(error, request)
                if attempt >= attempts - 1 or not _is_retryable_lm_error(normalized_error):
                    if normalized_error is error:
                        raise
                    raise normalized_error from error
                await _asleep_before_retry(attempt)
        raise RuntimeError("unreachable")

    def _forward_with_cache(self, request: LMRequest) -> LMResponse:
        if not _request_cache_enabled(request, self.cache):
            return self.forward(request)
        response = _cached_language_model_forward(
            cache_request=self._cache_request_for_mode(request, mode="sync"),
            lm=self,
            request=request,
        )
        return _prepare_cached_lm_response(response)

    async def _aforward_with_cache(self, request: LMRequest) -> LMResponse:
        if not _request_cache_enabled(request, self.cache):
            return await self.aforward(request)
        response = await _cached_language_model_aforward(
            cache_request=self._cache_request_for_mode(request, mode="async"),
            lm=self,
            request=request,
        )
        return _prepare_cached_lm_response(response)

    def _cache_request(self, request: LMRequest) -> dict[str, Any]:
        return {
            "lm_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "lm_state": _sanitize_cache_value(self.dump_state()),
            "request": _sanitize_cache_value(_model_dump_for_cache(request)),
        }

    def _cache_request_for_mode(self, request: LMRequest, *, mode: str) -> dict[str, Any]:
        cache_request = self._cache_request(request)
        cache_request["execution_mode"] = mode
        return cache_request

    def _cached_stream_events(self, request: LMRequest, *, mode: str) -> Iterator[LMStreamEvent] | None:
        if not _request_cache_enabled(request, self.cache):
            return None
        cached = _get_cached_lm_response(self._cache_request_for_mode(request, mode=mode))
        if cached is None:
            return None
        return _response_to_stream_events(_prepare_cached_lm_response(cached), model=request.model)

    def _cache_wrapped_stream_events(
        self,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        *,
        mode: str,
    ) -> Iterator[LMStreamEvent]:
        if not _request_cache_enabled(request, self.cache):
            yield from events
            return

        builder = _import_lm_types().LMOutputBuilder()
        response = None
        for event in events:
            built = builder.apply(event)
            if built is not None:
                response = built
                _put_cached_lm_response(self._cache_request_for_mode(request, mode=mode), response)
            yield event

    def _cached_astream_events(self, request: LMRequest, *, mode: str) -> AsyncIterator[LMStreamEvent] | None:
        if not _request_cache_enabled(request, self.cache):
            return None
        cached = _get_cached_lm_response(self._cache_request_for_mode(request, mode=mode))
        if cached is None:
            return None
        return _async_iter(_response_to_stream_events(_prepare_cached_lm_response(cached), model=request.model))

    async def _cache_wrapped_astream_events(
        self,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        *,
        mode: str,
    ) -> AsyncIterator[LMStreamEvent]:
        if not _request_cache_enabled(request, self.cache):
            async for event in events:
                yield event
            return

        builder = _import_lm_types().LMOutputBuilder()
        response = None
        async for event in events:
            built = builder.apply(event)
            if built is not None:
                response = built
                _put_cached_lm_response(self._cache_request_for_mode(request, mode=mode), response)
            yield event

    def _warn_zero_temp_rollout(self, request: LMRequest) -> None:
        cache = getattr(getattr(request, "config", None), "cache", None)
        rollout_id = getattr(cache, "rollout_id", None)
        temperature = getattr(getattr(request, "config", None), "temperature", None)
        if self._warned_zero_temp_rollout or rollout_id is None or temperature != 0:
            return
        warnings.warn(
            "rollout_id only affects DSPy's request cache when temperature=0; set temperature>0 "
            "to request a potentially different provider output.",
            UserWarning,
            stacklevel=3,
        )
        self._warned_zero_temp_rollout = True

    def _override_request(self, request: LMRequest, **kwargs: Any) -> LMRequest:
        if not kwargs:
            return request

        if hasattr(request, "with_config_overrides"):
            return request.with_config_overrides(**kwargs)

        if hasattr(request, "model_copy"):
            update = _request_config_update(request, kwargs)
            return request.model_copy(update=update, deep=True)

        raise TypeError("LMRequest overrides require with_config_overrides() or Pydantic model_copy().")

    def _require_stream_support(self, *, async_: bool) -> None:
        method_name = "aforward_stream" if async_ else "forward_stream"
        if self._method_overridden(method_name):
            return

        name = "async streaming" if async_ else "streaming"
        raise NotImplementedError(f"{type(self).__name__} does not support {name}; {method_name}() is not overridden.")

    def _finalize_response(self, request: LMRequest, response: LMResponse) -> LMResponse:
        self._track_usage(response)

        if not settings.disable_history:
            lm_types = _import_lm_types()
            entry = lm_types.LMHistoryEntry(
                request=request,
                response=response,
                timestamp=datetime.datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                model_type=getattr(self, "model_type", None),
            )
            self.update_history(entry)

        return response

    def _normalize_and_observe_error(self, error: Exception, request: LMRequest) -> Exception:
        return self.normalize_error(error, request)

    def _track_usage(self, response: LMResponse) -> None:
        if getattr(response, "cache_hit", False):
            return
        if not settings.usage_tracker:
            return

        usage = _response_usage_as_dict(response)
        if usage:
            settings.usage_tracker.add_usage(self.model, usage)

    def _method_overridden(self, method_name: str) -> bool:
        method = getattr(type(self), method_name, None)
        base_method = getattr(LanguageModel, method_name, None)
        return method is not None and base_method is not None and method is not base_method


def inspect_history(n: int = 1, file: Any | None = None) -> None:
    """Print recent interactions from all normalized language models."""
    pretty_print_history(GLOBAL_LANGUAGE_MODEL_HISTORY, n, file=file)


def _default_lm_kwargs(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build default request kwargs without storing omitted common values."""
    defaults = dict(kwargs)
    if temperature is not None:
        defaults["temperature"] = temperature
    if max_tokens is not None:
        defaults["max_tokens"] = max_tokens
    return defaults


def _prepare_cached_lm_response(response: Any) -> Any:
    if getattr(response, "cache_hit", False) and hasattr(response, "cost"):
        response.cost = None
    return response


def _is_retryable_lm_error(error: Exception) -> bool:
    try:
        from dspy.utils.exceptions import LMProviderError, LMRateLimitError
    except Exception:
        return False

    if isinstance(error, LMRateLimitError):
        return True
    if isinstance(error, LMProviderError):
        status = getattr(error, "status", None)
        return status is None or int(status) >= 500
    return False


def _sleep_before_retry(attempt: int) -> None:
    import time

    time.sleep(min(2 ** attempt, 8))


async def _asleep_before_retry(attempt: int) -> None:
    await anyio.sleep(min(2 ** attempt, 8))


def _cached_language_model_forward(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    def run(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
        return lm.forward(request)

    return run(cache_request=cache_request, lm=lm, request=request)


def _get_cached_lm_response(cache_request: dict[str, Any]) -> Any:
    import dspy

    return dspy.cache.get(_stream_cache_key(cache_request))


def _put_cached_lm_response(cache_request: dict[str, Any], response: Any) -> None:
    import dspy

    dspy.cache.put(_stream_cache_key(cache_request), response)


def _stream_cache_key(cache_request: dict[str, Any]) -> dict[str, Any]:
    return {**cache_request, "_fn_identifier": "dspy.LanguageModel.stream"}


async def _cached_language_model_aforward(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    async def run(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
        return await lm.aforward(request)

    return await run(cache_request=cache_request, lm=lm, request=request)


def _response_to_stream_events(response: Any, *, model: str | None = None) -> Iterator[Any]:
    lm_types = _import_lm_types()
    yield lm_types.LMStreamStartEvent(model=response.model or model)
    for output_index, output in enumerate(response.outputs):
        for part_index, part in enumerate(output.parts):
            delta = _part_to_stream_delta(part)
            if delta is not None:
                yield lm_types.LMStreamDeltaEvent(output_index=output_index, part_index=part_index, delta=delta)
        yield lm_types.LMStreamOutputEndEvent(
            output_index=output_index,
            finish_reason=output.finish_reason,
            truncated=output.truncated,
        )
    yield lm_types.LMStreamEndEvent(response=response)


def _part_to_stream_delta(part: Any) -> Any | None:
    lm_types = _import_lm_types()
    if isinstance(part, lm_types.LMTextPart):
        return lm_types.LMTextDelta(text=part.text)
    if isinstance(part, lm_types.LMThinkingPart):
        return lm_types.LMThinkingDelta(text=part.text)
    if isinstance(part, lm_types.LMToolCallPart):
        return lm_types.LMToolCallDelta(id=part.id, name=part.name, args_delta=json.dumps(part.args))
    if isinstance(part, lm_types.LMCitationPart):
        return lm_types.LMCitationDelta(citation=part)
    if isinstance(part, lm_types.LMImagePart):
        return lm_types.LMImageDelta(image=part)
    if isinstance(part, lm_types.LMAudioPart):
        return lm_types.LMAudioDelta(audio=part)
    return None


async def _async_iter(events: Iterator[Any]) -> AsyncIterator[Any]:
    for event in events:
        yield event


def _import_lm_types():
    try:
        from dspy.clients.language_models import types as lm_types
    except ImportError as exc:
        raise ImportError(
            "dspy.clients.language_models.types must be implemented before LanguageModel can "
            "normalize calls. Define LMRequest, LMResponse, and related types there."
        ) from exc
    return lm_types


def _request_cache_enabled(request: Any, default: bool) -> bool:
    config = getattr(request, "config", None)
    cache = getattr(config, "cache", None)
    enabled = getattr(cache, "enabled", None)
    return default if enabled is None else bool(enabled)


def _model_dump_for_cache(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _sanitize_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if key_text in {"api_key", "authorization", "x_api_key"}:
                continue
            sanitized[key] = _sanitize_cache_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_cache_value(item) for item in value]
    return value


def _request_config_update(request: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    config = getattr(request, "config", None)
    if config is None:
        return {"config": kwargs}
    if hasattr(config, "model_copy"):
        return {"config": config.model_copy(update=kwargs, deep=True)}
    if isinstance(config, dict):
        return {"config": {**config, **kwargs}}
    raise TypeError("Cannot override config on this LMRequest object.")


def _response_usage_as_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "usage_as_dict"):
        return response.usage_as_dict()

    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage)


def _sanitize_lm_request_for_callbacks(request: Any) -> Any:
    config = getattr(request, "config", None)
    if config is not None and hasattr(config, "model_copy"):
        config = config.model_copy(
            update={"extensions": _sanitize_callback_value(getattr(config, "extensions", {}) or {})},
            deep=True,
        )
        return request.model_copy(
            update={
                "config": config,
                "metadata": _sanitize_callback_value(getattr(request, "metadata", {}) or {}),
            },
            deep=True,
        )
    return _sanitize_callback_value(request)


def _sanitize_callback_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if key_text == "api_key" or key_text in {"authorization", "x_api_key"}:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_callback_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_callback_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_callback_value(item) for item in value]
    return value
