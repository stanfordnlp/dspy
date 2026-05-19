"""LiteLLM-backed normalized language model fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import anyio
import litellm
from litellm import ContextWindowExceededError as LiteLLMContextWindowExceededError

from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LMCapabilities
from dspy.clients.language_models.openai import completion_stream_to_events, responses_stream_to_events
from dspy.clients.language_models.openai_format import (
    completion_to_lm_response,
    responses_to_lm_response,
    to_openai_chat_request,
    to_openai_responses_request,
    to_openai_text_request,
)
from dspy.clients.language_models.types import LMRequest, LMResponse, LMStreamEvent

LiteLLMModelType = Literal["chat", "text", "responses"]


class LiteLLMLM(BaseLM):
    """Call any LiteLLM-supported provider through DSPy's normalized LM API.

    `LiteLLMLM` is DSPy's broad compatibility backend. Prefer native backends
    such as `OpenAIResponsesLM`, `OpenAIChatLM`, `OpenAITextLM`, `AnthropicLM`, or
    `GenAILM` when DSPy can route confidently. Use `LiteLLMLM` directly when a
    provider is supported by LiteLLM but does not yet have a normalized DSPy
    route.
    """

    def __init__(
        self,
        model: str,
        *,
        model_type: LiteLLMModelType = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ):
        if model_type not in {"chat", "text", "responses"}:
            raise ValueError("model_type must be 'chat', 'text', or 'responses'.")
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )
        self.model_type = model_type

    def get_capabilities(self) -> LMCapabilities:
        provider = self.model.split("/", 1)[0] if "/" in self.model else None
        try:
            function_calling = bool(litellm.supports_function_calling(model=self.model))
        except Exception:
            function_calling = False
        try:
            reasoning = bool(litellm.supports_reasoning(self.model))
        except Exception:
            reasoning = False
        try:
            response_schema = bool(
                litellm.supports_response_schema(model=self.model, custom_llm_provider=provider)
            )
        except Exception:
            response_schema = False
        return LMCapabilities(
            function_calling=function_calling,
            reasoning=reasoning,
            response_schema=response_schema,
            streaming=True,
            input_image=True,
            input_audio=True,
            input_file=True,
            tool_results=True,
        )

    def forward(self, request: LMRequest) -> LMResponse:
        if self.model_type == "responses":
            data = to_openai_responses_request(request)
            response = litellm.responses(
                **data,
                num_retries=self.num_retries,
                retry_strategy="exponential_backoff_retry",
                cache={"no-cache": True, "no-store": True},
            )
            return responses_to_lm_response(response, request)
        if self.model_type == "text":
            data = to_openai_text_request(request)
            response = litellm.text_completion(
                **data,
                num_retries=self.num_retries,
                retry_strategy="exponential_backoff_retry",
                cache={"no-cache": True, "no-store": True},
            )
            return completion_to_lm_response(response, request)

        data = to_openai_chat_request(request)
        response = litellm.completion(
            **data,
            num_retries=self.num_retries,
            retry_strategy="exponential_backoff_retry",
            cache={"no-cache": True, "no-store": True},
        )
        return completion_to_lm_response(response, request)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        if self.model_type == "responses":
            data = to_openai_responses_request(request)
            data["stream"] = True
            stream = litellm.responses(
                **data,
                num_retries=self.num_retries,
                retry_strategy="exponential_backoff_retry",
                cache={"no-cache": True, "no-store": True},
            )
            yield from responses_stream_to_events(stream, model=request.model)
            return

        data = to_openai_text_request(request) if self.model_type == "text" else to_openai_chat_request(request)
        data["stream"] = True
        if self.model_type == "chat":
            data.setdefault("stream_options", {"include_usage": True})
        completion = litellm.text_completion if self.model_type == "text" else litellm.completion
        stream = completion(
            **data,
            num_retries=self.num_retries,
            retry_strategy="exponential_backoff_retry",
            cache={"no-cache": True, "no-store": True},
        )
        yield from completion_stream_to_events(stream, model=request.model)

    async def aforward(self, request: LMRequest) -> LMResponse:
        return await anyio.to_thread.run_sync(self.forward, request)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        sentinel = object()
        iterator = iter(self.forward_stream(request))
        while True:
            event = await anyio.to_thread.run_sync(_next_or_sentinel, iterator, sentinel)
            if event is sentinel:
                break
            yield event

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        if isinstance(error, LiteLLMContextWindowExceededError):
            from dspy.utils.exceptions import ContextWindowExceededError

            provider = request.model.split("/", 1)[0] if "/" in request.model else "litellm"
            return ContextWindowExceededError(model=request.model, provider=provider)
        return error

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state["model_type"] = self.model_type
        return state


def _next_or_sentinel(iterator: Iterator[Any], sentinel: object) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return sentinel
