"""OpenRouter-backed DSPy LM wrappers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from openai import OpenAI

import dspy
from dr_dspy.lm_logging import PutEventFn, _LoggingMixin

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DSPY_ONLY_KWARGS = frozenset({"cache", "rollout_id"})

__all__ = [
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_BASE_URL",
    "LoggingOpenRouterLM",
    "OpenRouterLM",
]


class OpenRouterLM(dspy.BaseLM):
    """DSPy legacy LM that calls OpenRouter chat completions directly."""

    forward_contract = "legacy"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        reasoning: Mapping[str, Any] | None = None,
        client: OpenAI | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning = dict(reasoning or {})
        self._client = client

    @property
    def supported_params(self) -> set[str]:
        return {
            "max_completion_tokens",
            "max_tokens",
            "reasoning",
            "response_format",
            "seed",
            "tool_choice",
            "tools",
        }

    @property
    def supports_reasoning(self) -> bool:
        return True

    @property
    def supports_response_schema(self) -> bool:
        return True

    @property
    def supports_function_calling(self) -> bool:
        return True

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = self.api_key or os.getenv(OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise dspy.LMNotConfiguredError(
                f"{OPENROUTER_API_KEY_ENV} is not set",
                model=self.model,
                provider="openrouter",
            )

        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _request_kwargs(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        request_kwargs = {
            key: value
            for key, value in {**self.kwargs, **kwargs}.items()
            if key not in DSPY_ONLY_KWARGS and value is not None
        }
        if self.reasoning:
            extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
            extra_body["reasoning"] = dict(self.reasoning)
            request_kwargs["extra_body"] = extra_body
        return {"model": self.model, "messages": messages, **request_kwargs}

    def forward(
        self,
        prompt: Any = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        messages = messages or [{"role": "user", "content": prompt}]
        return self._get_client().chat.completions.create(
            **self._request_kwargs(messages, kwargs)
        )

    async def aforward(
        self,
        prompt: Any = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.forward(prompt=prompt, messages=messages, **kwargs)


class LoggingOpenRouterLM(_LoggingMixin, OpenRouterLM):
    """OpenRouterLM with lm.request/response/error logging."""

    def __init__(self, model: str, *, log: PutEventFn, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._log = log

    def forward(
        self,
        prompt: Any = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._run_logged_forward(
            lambda: super(LoggingOpenRouterLM, self).forward(
                prompt=prompt, messages=messages, **kwargs
            ),
            messages=messages,
            kwargs=kwargs,
        )

    async def aforward(
        self,
        prompt: Any = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.forward(prompt=prompt, messages=messages, **kwargs)
