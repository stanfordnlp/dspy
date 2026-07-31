"""Typed LM engine for OpenAI Chat Completions-compatible HTTP endpoints.

`_OpenAICompatLM` is an internal engine, not a public API: users configure
`dspy.LM`, and the router constructs an engine under the hood. It uses direct
`requests` transport and DSPy's typed `LMRequest` / `LMResponse` boundary.
Async calls run the same synchronous transport in an AnyIO worker thread.
Streaming is supported through `forward_stream`, which parses the endpoint's
SSE response into normalized `LMStreamEvent`s.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

import anyio
import requests

import dspy
from dspy.clients.base_lm import (
    LM_CLASS_STATE_KEY,
    SENSITIVE_HEADER_NAMES,
    BaseLM,
    _scrub_lm_state_on_pickle,
)
from dspy.clients.openai_format import (
    ChatCompletionChunkAssembler,
    completion_to_lm_response,
    to_openai_chat_request,
)
from dspy.core.types import (
    LMOutputBuilder,
    LMRequest,
    LMResponse,
    LMStreamEvent,
    LMStreamStartEvent,
    response_to_stream_events,
)
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMAuthError,
    LMBillingError,
    LMConfigurationError,
    LMError,
    LMInvalidRequestError,
    LMNotConfiguredError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
    LMUnsupportedModelError,
    is_retryable_lm_error,
)

__all__ = ["_OpenAICompatLM"]

logger = logging.getLogger(__name__)

_PROVIDER_CODE_MAP: dict[str, type[LMError]] = {
    "context_length_exceeded": ContextWindowExceededError,
    "model_not_found": LMUnsupportedModelError,
    "model_not_available": LMUnsupportedModelError,
    "unsupported_model": LMUnsupportedModelError,
    "insufficient_quota": LMBillingError,
    "invalid_api_key": LMAuthError,
    "authentication_error": LMAuthError,
    "rate_limit_exceeded": LMRateLimitError,
    "rate_limit_error": LMRateLimitError,
}
_MODEL_ERROR_CODES = frozenset({"model_not_found", "model_not_available", "unsupported_model"})
_SENSITIVE_HEADER_NAMES = SENSITIVE_HEADER_NAMES


@dataclass(frozen=True)
class _RawResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


_Post = Callable[[str, dict[str, Any], dict[str, str], float], _RawResponse]


def _requests_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> _RawResponse:
    response = requests.post(url, json=body, headers=headers, timeout=timeout)
    return _RawResponse(status=response.status_code, headers=dict(response.headers), body=response.content)


def _header(headers: Mapping[str, str] | None, *names: str) -> str | None:
    if not headers:
        return None
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return next((lowered[name.lower()] for name in names if name.lower() in lowered), None)


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    value = _header(headers, "retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_model_error(message: str, *codes: str) -> bool:
    lowered = " ".join(str(value) for value in (message, *codes) if value).lower()
    return "model" in lowered and any(
        marker in lowered
        for marker in (
            "not found",
            "does not exist",
            "not exist",
            "not supported",
            "unsupported",
            "not available",
            "unknown",
        )
    )


def _normalize_error(
    status: int,
    body: str,
    *,
    headers: Mapping[str, str] | None = None,
    model: str | None = None,
    provider: str = "openai_compat",
) -> LMError:
    """Map an OpenAI-shaped HTTP error response to a structured DSPy error."""
    message: str | None = None
    provider_code: str | None = None
    error_type: str | None = None

    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        data = None

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            raw_message = error.get("message")
            message = str(raw_message) if raw_message is not None else None
            raw_code = error.get("code")
            raw_type = error.get("type")
            provider_code = str(raw_code) if raw_code not in (None, "") else None
            error_type = str(raw_type) if raw_type not in (None, "") else None
        elif error is not None:
            message = str(error)

    if message is None:
        message = body.strip()[:500] if body and body.strip() else f"HTTP {status}"

    metadata = {
        "model": model,
        "provider": provider,
        "provider_code": provider_code or error_type,
        "status": status,
        "request_id": _header(headers, "x-request-id", "request-id", "x-amzn-requestid", "x-ms-request-id"),
        "retry_after": _retry_after(headers),
    }

    error_class = next(
        (_PROVIDER_CODE_MAP[value] for value in (provider_code, error_type) if value in _PROVIDER_CODE_MAP),
        None,
    )
    if error_class is not None:
        return error_class(message=message, **metadata)

    if status == 404 and _is_model_error(message, provider_code or "", error_type or ""):
        return LMUnsupportedModelError(message=message, **metadata)

    return _error_class_from_status(status)(message=message, **metadata)


def _error_class_from_status(status: int) -> type[LMError]:
    if status in (401, 403):
        return LMAuthError
    if status == 402:
        return LMBillingError
    if status in (408, 504):
        return LMTimeoutError
    if status == 429:
        return LMRateLimitError
    if 400 <= status < 500:
        return LMInvalidRequestError
    if status >= 500:
        return LMServerError
    return LMProviderError


class _OpenAICompatLM(BaseLM):
    """Internal typed LM engine for an OpenAI Chat Completions-compatible endpoint.

    This engine speaks the Chat Completions wire format over direct HTTP with
    no LiteLLM or OpenAI SDK dependency — for servers such as vLLM, SGLang,
    Ollama, llama.cpp, or a gateway. It is not a public API: `dspy.LM` is the
    user-facing interface, and serialized state always round-trips through
    `dspy.LM`, never through this class.

    Example (internal construction):
        ```python
        from dspy.clients.openai_compat_lm import _OpenAICompatLM

        lm = _OpenAICompatLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
        )
        ```

    Args:
        model: Model identifier accepted by the endpoint.
        base_url: API base URL, such as `http://localhost:8000/v1`.
        api_key: Optional explicit bearer token, or a zero-argument callable
            returning one. A callable is invoked on every request, so vaults,
            OAuth refreshers, and rotating credentials plug in without the LM
            knowing which; the resolved token is only ever placed in the
            `Authorization` header. Omit for local endpoints that need no key.
        api_key_env: Optional environment variable checked when `api_key` is absent.
        use_openai_api_key_env: Whether to fall back to `OPENAI_API_KEY`. Off by
            default so a key meant for OpenAI is never sent to another endpoint
            without an explicit opt-in.
        require_auth: When True, raise `dspy.LMNotConfiguredError` locally if
            no credential resolves, instead of sending an unauthenticated
            request and waiting for the endpoint to reject it. Leave False for
            local endpoints that need no key.
        timeout: Request timeout in seconds.
        extra_headers: Additional HTTP headers. Explicit values override defaults.
        supports_function_calling: Opt into native tool calls.
        supports_reasoning: Opt into native reasoning.
        supports_response_schema: Opt into structured response schemas.
        model_type: Must be `"chat"`; this client speaks Chat Completions only.
        temperature: Default sampling temperature for every request.
        max_tokens: Default response token limit for every request.
        cache: Whether to cache responses (on by default). Repeating an
            identical request returns the cached response without an HTTP call.
        callbacks: Callback handlers observing each LM call.
        num_retries: How many times to retry a failed request when the error
            is retryable, such as a rate limit or a server error.
        **kwargs: Default request parameters, such as `top_p` or `stop`,
            applied to every request unless overridden at call time.

    When the capability flags are False, DSPy's adapters express tools,
    reasoning, and structured output through prompting instead of the native
    API. When a flag is True but the server does not actually support the
    feature, requests fail with a provider error.

    API keys are never serialized. If an explicit key was used, serialized state
    disables ambient `OPENAI_API_KEY` fallback so loading cannot silently
    switch to another account.
    """

    forward_contract = "typed_lm"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | Callable[[], str] | None = None,
        api_key_env: str | None = None,
        use_openai_api_key_env: bool = False,
        require_auth: bool = False,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        supports_function_calling: bool = False,
        supports_reasoning: bool = False,
        supports_response_schema: bool = False,
        model_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        _post: _Post | None = None,
        **kwargs: Any,
    ):
        if model_type != "chat":
            raise LMConfigurationError(
                f"OpenAICompatLM only supports model_type='chat', but got {model_type!r}.",
                model=model,
                provider="openai_compat",
            )
        if not isinstance(base_url, str) or not base_url.strip():
            raise LMConfigurationError(
                "OpenAICompatLM requires a non-empty base_url.", model=model, provider="openai_compat"
            )
        if timeout <= 0:
            raise LMConfigurationError(
                "OpenAICompatLM timeout must be greater than zero.", model=model, provider="openai_compat"
            )
        if num_retries < 0:
            raise LMConfigurationError(
                "OpenAICompatLM num_retries cannot be negative.", model=model, provider="openai_compat"
            )

        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.use_openai_api_key_env = bool(use_openai_api_key_env)
        self.require_auth = bool(require_auth)
        self.timeout = float(timeout)
        self.extra_headers = {str(key): str(value) for key, value in (extra_headers or {}).items()}
        self._supports_function_calling = bool(supports_function_calling)
        self._supports_reasoning = bool(supports_reasoning)
        self._supports_response_schema = bool(supports_response_schema)
        self._api_key = api_key
        self._post = _post or _requests_post

    def __getstate__(self):
        state = super().__getstate__()
        if not _scrub_lm_state_on_pickle.get():
            return state

        # Engine-specific hygiene on top of BaseLM's: the credential (string
        # or callable), sensitive headers, and the transport callable stay out
        # of program artifacts.
        state["_api_key"] = None
        state["extra_headers"] = {
            key: value for key, value in self.extra_headers.items() if key.lower() not in _SENSITIVE_HEADER_NAMES
        }
        state["_post"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.__dict__.get("_post") is None:
            self._post = _requests_post

    def _resolved_api_key(self) -> str | None:
        if self._api_key is not None:
            return self._api_key() if callable(self._api_key) else self._api_key
        if self.api_key_env is not None:
            value = os.environ.get(self.api_key_env)
            if value:
                return value
        if self.use_openai_api_key_env:
            return os.environ.get("OPENAI_API_KEY")
        return None

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": f"DSPy/{dspy.__version__}"}
        api_key = self._resolved_api_key()
        if api_key is None and self.require_auth:
            fixes = [
                "pass api_key= to OpenAICompatLM",
                f"set {self.api_key_env}" if self.api_key_env else None,
                "set OPENAI_API_KEY with use_openai_api_key_env=True" if self.use_openai_api_key_env else None,
            ]
            raise LMNotConfiguredError(
                "OpenAICompatLM requires a credential (require_auth=True) but none resolved. "
                f"To fix: {'; or '.join(fix for fix in fixes if fix)}.",
                model=self.model,
                provider="openai_compat",
            )
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.extra_headers)
        return headers

    def _cache_request(self, request: LMRequest) -> dict[str, Any]:
        # Key on the OpenAI-shaped wire payload rather than the LMRequest model:
        # config values such as a pydantic `response_format` class are only
        # JSON-serializable after `to_openai_chat_request()` maps them, and the
        # payload naturally excludes DSPy-only cache config.
        api_key = self._resolved_api_key()
        header_fingerprints = {
            name.lower(): hashlib.sha256(value.encode()).hexdigest()
            for name, value in sorted(self.extra_headers.items(), key=lambda item: item[0].lower())
        }
        cache_config = request.config.cache
        return {
            "_fn_identifier": "dspy.clients.openai_compat_lm._OpenAICompatLM.forward",
            "base_url": self.base_url,
            "request": to_openai_chat_request(request),
            "rollout_id": cache_config.rollout_id if cache_config is not None else None,
            "credential_fingerprint": hashlib.sha256(api_key.encode()).hexdigest() if api_key else None,
            "header_fingerprints": header_fingerprints,
        }

    def _cache_enabled(self, request: LMRequest) -> bool:
        config = request.config.cache
        if config is None or config.enabled is None:
            return self.cache
        return config.enabled

    def _request_once(self, request: LMRequest) -> LMResponse:
        payload = to_openai_chat_request(request)
        try:
            raw = self._post(f"{self.base_url}/chat/completions", payload, self._request_headers(), self.timeout)
        except requests.Timeout as error:
            raise LMTimeoutError(
                str(error) or "LM request timed out", model=self.model, provider="openai_compat"
            ) from error
        except requests.RequestException as error:
            raise LMTransportError(
                str(error) or "LM transport failed", model=self.model, provider="openai_compat"
            ) from error
        except TimeoutError as error:
            raise LMTimeoutError(
                str(error) or "LM request timed out", model=self.model, provider="openai_compat"
            ) from error
        except OSError as error:
            raise LMTransportError(
                str(error) or "LM transport failed", model=self.model, provider="openai_compat"
            ) from error

        body = raw.body.decode("utf-8", errors="replace")
        if raw.status >= 400:
            raise _normalize_error(raw.status, body, headers=raw.headers, model=self.model)
        try:
            provider_response = json.loads(body)
        except (TypeError, ValueError) as error:
            raise LMProviderError(
                "OpenAI-compatible endpoint returned a non-JSON success response.",
                model=self.model,
                provider="openai_compat",
                status=raw.status,
                request_id=_header(raw.headers, "x-request-id", "request-id"),
            ) from error
        if not isinstance(provider_response, dict):
            raise LMProviderError(
                "OpenAI-compatible endpoint returned a JSON response that was not an object.",
                model=self.model,
                provider="openai_compat",
                status=raw.status,
            )

        try:
            response = completion_to_lm_response(provider_response, request)
        except Exception as error:
            raise LMProviderError(
                "OpenAI-compatible endpoint returned an invalid Chat Completions response.",
                model=self.model,
                provider="openai_compat",
                status=raw.status,
                request_id=_header(raw.headers, "x-request-id", "request-id"),
            ) from error
        self._warn_on_truncation(response, request)
        return response

    def _warn_on_truncation(self, response: LMResponse, request: LMRequest) -> None:
        if any(output.truncated for output in response.outputs):
            logger.warning(
                "OpenAICompatLM response was truncated (finish_reason='length', max_tokens=%s). "
                "Inspect recent LM calls with `dspy.inspect_history()`, or pass a larger max_tokens.",
                request.config.max_tokens,
            )

    def _request_with_retries(self, request: LMRequest) -> LMResponse:
        for attempt in range(self.num_retries + 1):
            try:
                return self._request_once(request)
            except LMError as error:
                if attempt >= self.num_retries or not is_retryable_lm_error(error):
                    raise
                delay = error.retry_after if error.retry_after is not None else min(2**attempt, 60)
                time.sleep(max(delay, 0.0))
        raise AssertionError("retry loop did not return or raise")

    def forward(self, request: LMRequest) -> LMResponse:
        """Call the endpoint synchronously and return a normalized response."""
        if not self._cache_enabled(request):
            return self._request_with_retries(request)

        cache_request = self._cache_request(request)
        cached = dspy.cache.get(cache_request)
        if cached is not None:
            return cached
        response = self._request_with_retries(request)
        dspy.cache.put(cache_request, response)
        return response

    async def aforward(self, request: LMRequest) -> LMResponse:
        """Async version of `forward`; runs the same request in a background thread."""
        return await anyio.to_thread.run_sync(self.forward, request)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        """Stream the endpoint's SSE response as normalized events.

        Caching matches `forward`: a cache hit replays the stored response as
        events without an HTTP call, and a completed live stream stores its
        final response under the same cache key — so a streamed call and a
        buffered call of the same request share one cache entry.
        """
        cache_request = self._cache_request(request) if self._cache_enabled(request) else None
        if cache_request is not None:
            cached = dspy.cache.get(cache_request)
            if cached is not None:
                yield from response_to_stream_events(cached)
                return

        builder = LMOutputBuilder()
        response: LMResponse | None = None
        for event in self._stream_with_retries(request):
            response = builder.apply(event) or response
            yield event

        if response is not None:
            self._warn_on_truncation(response, request)
            if cache_request is not None:
                dspy.cache.put(cache_request, response)

    def _stream_with_retries(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        # Retry only failures that happen before the first event is yielded;
        # a stream that already produced output cannot be transparently
        # restarted without replaying partial content to the consumer.
        for attempt in range(self.num_retries + 1):
            started = False
            try:
                for event in self._stream_once(request):
                    started = True
                    yield event
                return
            except LMError as error:
                if started or attempt >= self.num_retries or not is_retryable_lm_error(error):
                    raise
                delay = error.retry_after if error.retry_after is not None else min(2**attempt, 60)
                time.sleep(max(delay, 0.0))
        raise AssertionError("retry loop did not return or raise")

    def _stream_once(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        payload = to_openai_chat_request(request)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})
        headers = self._request_headers()
        try:
            http_response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True,
            )
        except requests.Timeout as error:
            raise LMTimeoutError(
                str(error) or "LM request timed out", model=self.model, provider="openai_compat"
            ) from error
        except requests.RequestException as error:
            raise LMTransportError(
                str(error) or "LM transport failed", model=self.model, provider="openai_compat"
            ) from error

        with http_response:
            if http_response.status_code >= 400:
                body = http_response.content.decode("utf-8", errors="replace")
                raise _normalize_error(
                    http_response.status_code, body, headers=http_response.headers, model=self.model
                )

            yield LMStreamStartEvent(model=request.model)
            assembler = ChatCompletionChunkAssembler()
            try:
                for line in http_response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError as error:
                        raise LMProviderError(
                            "OpenAI-compatible endpoint sent a non-JSON stream chunk.",
                            model=self.model,
                            provider="openai_compat",
                            status=http_response.status_code,
                            request_id=_header(http_response.headers, "x-request-id", "request-id"),
                        ) from error
                    yield from assembler.events(chunk)
            except requests.Timeout as error:
                raise LMTimeoutError(
                    str(error) or "LM stream timed out", model=self.model, provider="openai_compat"
                ) from error
            except requests.RequestException as error:
                raise LMTransportError(
                    str(error) or "LM stream transport failed", model=self.model, provider="openai_compat"
                ) from error
            yield from assembler.end_events()

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_function_calling(self) -> bool:
        return self._supports_function_calling

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def supports_response_schema(self) -> bool:
        return self._supports_response_schema

    @property
    def supported_params(self) -> set[str]:
        params = {"temperature", "max_tokens", "top_p", "stop", "n", "logprobs"}
        if self.supports_function_calling:
            params.update({"tools", "tool_choice", "parallel_tool_calls"})
        if self.supports_reasoning:
            params.add("reasoning_effort")
        if self.supports_response_schema:
            params.add("response_format")
        return params

    def dump_state(self) -> dict[str, Any]:
        """Serialize as `dspy.LM` router state, never as engine state.

        Saved programs record the router-level facts — a LiteLLM-routable model
        string, `api_base`, and default request parameters — so
        `BaseLM.load_state` reconstructs a `dspy.LM` pointed at the same
        endpoint. Engine-only configuration (credential resolution, declared
        capabilities, the raw endpoint model id) is carried losslessly under
        the `"engine"` key, which `dspy.LM` preserves for the router without
        interpreting today.
        """
        request_kwargs = {
            key: value for key, value in self.kwargs.items() if key not in ("api_key", LM_CLASS_STATE_KEY)
        }
        safe_headers = {
            key: value for key, value in self.extra_headers.items() if key.lower() not in _SENSITIVE_HEADER_NAMES
        }
        state = {
            LM_CLASS_STATE_KEY: "dspy.clients.lm.LM",
            # LiteLLM routes `openai/<model>` + `api_base` to the same
            # Chat Completions endpoint this engine speaks to directly.
            "model": f"openai/{self.model}",
            "model_type": self.model_type,
            "cache": self.cache,
            "num_retries": self.num_retries,
            **request_kwargs,
            "api_base": self.base_url,
            "timeout": self.timeout,
            "engine": {
                "engine": "openai_compat",
                "model": self.model,
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "use_openai_api_key_env": self.use_openai_api_key_env if self._api_key is None else False,
                "require_auth": self.require_auth,
                "timeout": self.timeout,
                "extra_headers": safe_headers,
                "supports_function_calling": self._supports_function_calling,
                "supports_reasoning": self._supports_reasoning,
                "supports_response_schema": self._supports_response_schema,
            },
        }
        if safe_headers:
            state["extra_headers"] = safe_headers
        return state

    # Router-level state keys that duplicate or do not apply to this engine's
    # constructor; everything else in the router state is a request default.
    _ROUTER_ONLY_STATE_KEYS = frozenset(
        {
            LM_CLASS_STATE_KEY,
            "model",
            "model_type",
            "api_base",
            "timeout",
            "extra_headers",
            "finetuning_model",
            "launch_kwargs",
            "train_kwargs",
            "use_developer_role",
        }
    )

    @classmethod
    def _from_router_state(cls, engine_state: dict[str, Any], router_state: dict[str, Any]) -> _OpenAICompatLM:
        """Reconstruct the engine from a serialized `engine` block plus router state.

        The inverse of `dump_state`: the `engine` block carries this engine's
        constructor configuration, and the surrounding router state contributes
        the shared request defaults (`cache`, `num_retries`, `temperature`,
        `max_tokens`, extra request parameters).
        """
        engine_kwargs = {key: value for key, value in engine_state.items() if key != "engine"}
        request_kwargs = {
            key: value for key, value in router_state.items() if key not in cls._ROUTER_ONLY_STATE_KEYS
        }
        return cls(**engine_kwargs, **request_kwargs)
