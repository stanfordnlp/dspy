from __future__ import annotations

import asyncio
from typing import Any

from dr_dspy.openrouter_lm import (
    OPENROUTER_BASE_URL,
    LoggingOpenRouterLM,
    OpenRouterLM,
)
from dspy.utils.dummies import dotdict  # type: ignore[attr-defined]


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return dotdict(
            model=kwargs["model"],
            choices=[
                dotdict(
                    message=dotdict(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=dotdict(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = dotdict(completions=FakeCompletions())


def test_openrouter_lm_sends_reasoning_from_configuration() -> None:
    client = FakeClient()
    lm = OpenRouterLM(
        "openai/gpt-5-nano",
        client=client,
        reasoning={"effort": "minimal", "exclude": False},
        cache=False,
        temperature=None,
    )

    lm.forward(
        messages=[{"role": "user", "content": "hello"}],
        cache=True,
        rollout_id=10,
        max_tokens=32,
    )

    assert client.chat.completions.calls == [
        {
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "extra_body": {
                "reasoning": {"effort": "minimal", "exclude": False}
            },
        }
    ]


def test_openrouter_lm_works_through_dspy_legacy_call_path() -> None:
    lm = OpenRouterLM("openai/gpt-5-nano", client=FakeClient())

    assert lm("hello") == ["ok"]


def test_openrouter_lm_uses_json_mode_not_structured_outputs() -> None:
    lm = OpenRouterLM("openai/gpt-5-nano", client=FakeClient())

    assert "response_format" in lm.supported_params
    assert lm.supports_response_schema is False


def test_openrouter_lm_uses_openrouter_client_configuration(
    monkeypatch,
) -> None:
    constructed: list[dict[str, Any]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)
            self.chat = dotdict(completions=FakeCompletions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("dr_dspy.openrouter_lm.OpenAI", FakeOpenAI)

    lm = OpenRouterLM("openai/gpt-5-nano")
    lm.forward(prompt="hello")

    assert constructed == [
        {"api_key": "test-key", "base_url": OPENROUTER_BASE_URL}
    ]


def test_logging_openrouter_lm_emits_request_and_response_events() -> None:
    client = FakeClient()
    events: list[dict[str, Any]] = []
    lm = LoggingOpenRouterLM(
        "openai/gpt-5-nano",
        client=client,
        log=lambda event_type, **kwargs: events.append(
            {"event_type": event_type, **kwargs}
        ),
    )

    lm.forward(messages=[{"role": "user", "content": "hello"}])

    event_types = [event["event_type"] for event in events]
    assert event_types == ["lm.request", "lm.response"]
    assert events[0]["payload"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert events[0]["payload"]["req_id"] == events[1]["payload"]["req_id"]
    assert isinstance(events[1]["payload"]["dt"], float)


def test_logging_openrouter_lm_aforward_uses_sync_forward(
    monkeypatch,
) -> None:
    lm = LoggingOpenRouterLM(
        "openai/gpt-5-nano",
        client=FakeClient(),
        log=lambda *_args, **_kwargs: None,
    )
    calls: list[dict[str, Any]] = []

    def fake_forward(
        prompt: Any = None,
        messages: Any = None,
        **kwargs: Any,
    ) -> str:
        calls.append(
            {"prompt": prompt, "messages": messages, "kwargs": dict(kwargs)}
        )
        return "sync-result"

    async def unexpected_openrouter_async_path(
        _self: OpenRouterLM,
        prompt: Any = None,
        messages: Any = None,
        **kwargs: Any,
    ) -> str:
        raise AssertionError(
            "LoggingOpenRouterLM.aforward reached OpenRouterLM.aforward"
        )

    monkeypatch.setattr(lm, "forward", fake_forward)
    monkeypatch.setattr(
        OpenRouterLM, "aforward", unexpected_openrouter_async_path
    )

    result = asyncio.run(
        lm.aforward(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
        )
    )

    assert result == "sync-result"
    assert calls == [
        {
            "prompt": None,
            "messages": [{"role": "user", "content": "hello"}],
            "kwargs": {"temperature": 0.0},
        }
    ]
