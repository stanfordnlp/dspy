"""Small lm15 engines for tests: no provider SDK objects, no network."""

from dspy.lm15 import Message, Response, TextPart, ThinkingPart, ToolCallPart, Usage, response_to_events


def make_response(text="", *, model="test", tool_calls=(), thinking=None, finish_reason=None, usage=None, **extra):
    """Build a Response from plain values.

    `tool_calls` is a list of (id, name, input) tuples or dicts with those keys.
    """
    parts = []
    if thinking:
        parts.append(ThinkingPart(thinking))
    if text:
        parts.append(TextPart(text))
    for call in tool_calls:
        if isinstance(call, dict):
            parts.append(ToolCallPart(id=call.get("id", "call_1"), name=call["name"], input=call.get("input", {})))
        else:
            parts.append(ToolCallPart(id=call[0], name=call[1], input=call[2]))
    if not parts:
        parts.append(TextPart(""))
    if finish_reason is None:
        finish_reason = "tool_call" if tool_calls else "stop"
    usage = usage or Usage(input_tokens=1, output_tokens=1, total_tokens=2)
    return Response(id=None, model=model, message=Message.assistant(parts), finish_reason=finish_reason,
                    usage=usage, **extra)


class RecordingEngine:
    """Answer each request with the next scripted Response, recording every Request."""

    def __init__(self, responses=None, *, stream=True):
        self.responses = list(responses or [])
        self.requests = []
        self.streams = 0
        self._stream = stream
        self.supports_function_calling = True
        self.supports_reasoning = True
        self.supports_response_schema = True
        self.supported_params = {"temperature", "max_tokens", "tools", "tool_choice", "parallel_tool_calls",
                                 "response_format", "reasoning_effort", "logprobs", "top_logprobs", "stop", "n"}

    def _next(self, request):
        self.requests.append(request)
        if not self.responses:
            return make_response("", model=request.model)
        response = self.responses.pop(0)
        if isinstance(response, str):
            response = make_response(response, model=request.model)
        if isinstance(response, Exception):
            raise response
        return response

    def complete(self, request):
        return self._next(request)

    def stream(self, request):
        if not self._stream:
            raise NotImplementedError
        self.streams += 1
        return response_to_events(self._next(request))

    def close(self):
        pass


class AsyncRecordingEngine:
    def __init__(self, sync):
        self.sync = sync

    async def complete(self, request):
        return self.sync.complete(request)

    async def stream(self, request):
        for event in self.sync.stream(request):
            yield event

    async def aclose(self):
        pass


def recording_lm(responses=None, *, model="test/model", **kwargs):
    """A dspy.LM bound to a RecordingEngine; `lm.engine.requests` holds what was sent."""
    import dspy

    engine = RecordingEngine(responses)
    kwargs.setdefault("cache", False)
    return dspy.LM(model, engine=engine, async_engine=AsyncRecordingEngine(engine), **kwargs)


def litellm_response(text="", *, model="openai/gpt-4o-mini", tool_calls=None, usage=None):
    """A minimal valid LiteLLM ModelResponse for tests that mock `litellm.completion`."""
    from litellm import Choices, Message, ModelResponse

    message = Message(content=text, tool_calls=tool_calls)
    return ModelResponse(choices=[Choices(message=message, finish_reason="stop")], model=model, usage=usage)


def _has_finish_reason(chunk):
    choices = getattr(chunk, "choices", None) or (chunk.get("choices") if isinstance(chunk, dict) else None) or []
    for choice in choices:
        reason = getattr(choice, "finish_reason", None) if not isinstance(choice, dict) else choice.get("finish_reason")
        if reason is not None:
            return True
    return False


def _terminal_chunk():
    from litellm import ModelResponseStream
    from litellm.types.utils import Delta, StreamingChoices

    return ModelResponseStream(choices=[StreamingChoices(delta=Delta(), finish_reason="stop")])


async def _terminated(source):
    """Recorded fixtures often omit the final chunk a real LiteLLM stream sends."""
    import inspect

    if inspect.isawaitable(source):
        source = await source
    last = None
    async for chunk in source:
        last = chunk
        yield chunk
    if last is not None and not _has_finish_reason(last):
        yield _terminal_chunk()


def _materialize_stream(value):
    """Turn what a patched `litellm.acompletion` returned into a sync chunk iterator."""
    import asyncio

    async def collect(source):
        return [chunk async for chunk in _terminated(source)]

    return iter(asyncio.run(collect(value)))


def patch_litellm_streaming(**kwargs):
    """Patch `litellm.acompletion` as given, and `litellm.completion` with a sync view of it.

    Streaming tests describe chunks as async generators. A synchronous DSPy
    program streams through `litellm.completion(stream=True)` in a worker
    thread, so the same recorded chunks are replayed there as a plain iterator.
    """
    from contextlib import contextmanager
    from unittest import mock

    @contextmanager
    def patched():
        with mock.patch("litellm.acompletion", **kwargs) as recorded:
            def completion(*args, **kw):
                return _materialize_stream(recorded(*args, **kw))

            async def acompletion(*args, **kw):
                return _terminated(recorded(*args, **kw))

            with mock.patch("litellm.completion", side_effect=completion), \
                 mock.patch("litellm.acompletion", side_effect=acompletion):
                yield recorded

    return patched()


def spy_lm(model="dummy", replies=("",), **kwargs):
    """A `dspy.LM` that answers with fixed text and records what adapters sent.

    `lm.calls[i]["messages"]` is the OpenAI-shaped view of the i-th Request
    (system first), which keeps prompt-content assertions readable;
    `lm.calls[i]["request"]` is the Request itself.
    """
    import dspy
    from dspy.clients.lm15_boundary import request_kwargs

    class ReplyEngine(RecordingEngine):
        def _next(self, request):
            self.requests.append(request)
            reply = replies[min(len(self.requests) - 1, len(replies) - 1)]
            return make_response(reply, model=request.model)

    class SpyLM(dspy.LM):
        @property
        def calls(self):
            return [{"messages": request_kwargs(r, "chat")["messages"], "request": r} for r in self.engine.requests]

    engine = ReplyEngine()
    return SpyLM(model, engine=engine, async_engine=AsyncRecordingEngine(engine), cache=False, **kwargs)


def failing_lm(error, *, model="dummy"):
    """A `dspy.LM` whose engine raises `error` on every call."""
    import dspy

    class FailingEngine:
        def complete(self, request):
            raise error

    return dspy.LM(model, engine=FailingEngine(), cache=False, num_retries=0)
