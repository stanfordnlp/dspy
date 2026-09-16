"""Shared adapter test helpers.

Adapters build one lm15 `Request` per call. The exact-format tests below
capture that Request and look at it through lm15's own Chat Completions
writer, so assertions read as the messages and options an OpenAI-compatible
server would receive. `format_request` returns the Request itself for tests
about lm15 parts, tools and config.
"""

import dspy
from dspy.clients.lm15_boundary import request_kwargs
from dspy.lm15 import Request
from tests.test_utils.engines import AsyncRecordingEngine, RecordingEngine


class StopAdapterCallCapture(BaseException):
    """Stop adapter execution after capturing the LM call.

    Raising here avoids needing to craft a parseable LM response for every
    signature under test.
    """


class CapturingEngine(RecordingEngine):
    def complete(self, request):
        self.requests.append(request)
        raise StopAdapterCallCapture

    def stream(self, request):
        self.requests.append(request)
        raise StopAdapterCallCapture


class CapturingLM(dspy.BaseLM):
    """Record the Request an adapter builds, with the capabilities of `source_lm`."""

    def __init__(self, source_lm=None):
        engine = CapturingEngine()
        # Capabilities and extra options come from the source LM; its sampling
        # defaults do not, so assertions see only what the adapter produced.
        kwargs = {k: v for k, v in (source_lm.kwargs if source_lm is not None else {}).items()
                  if k not in ("temperature", "max_tokens", "max_completion_tokens")}
        super().__init__(
            model=source_lm.model if source_lm is not None else "dummy",
            model_type=source_lm.model_type if source_lm is not None else "chat",
            cache=False,
            engine=engine,
            async_engine=AsyncRecordingEngine(engine),
            **kwargs,
        )
        self.source_lm = source_lm or dspy.utils.DummyLM([{}])

    @property
    def calls(self):
        return self.engine.requests

    @property
    def supports_function_calling(self):
        return self.source_lm.supports_function_calling

    @property
    def supports_reasoning(self):
        return self.source_lm.supports_reasoning

    @property
    def supports_response_schema(self):
        return self.source_lm.supports_response_schema

    @property
    def supported_params(self):
        return self.source_lm.supported_params


def format_request(adapter, signature, demos, inputs, lm_kwargs=None, lm=None) -> Request:
    capturing_lm = CapturingLM(lm)
    try:
        adapter(capturing_lm, dict(lm_kwargs or {}), signature, demos, inputs)
    except StopAdapterCallCapture:
        pass

    assert len(capturing_lm.calls) == 1
    return capturing_lm.calls[0]


def openai_view(request: Request) -> tuple[list[dict], dict]:
    """(messages, kwargs) as lm15 writes this Request for a Chat Completions server."""
    data = request_kwargs(request, "chat")
    messages = data.pop("messages")
    if "max_completion_tokens" in data:
        data["max_tokens"] = data.pop("max_completion_tokens")
    return messages, data


def format_messages_and_lm_kwargs(adapter, signature, demos, inputs, lm_kwargs=None, lm=None):
    return openai_view(format_request(adapter, signature, demos, inputs, lm_kwargs, lm))
