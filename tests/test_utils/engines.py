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


def patch_streaming(**kwargs):
    """Replay recorded OpenAI chunks or canonical events at the engine boundary.

    The native parser reads plain wire dictionaries, without provider SDK
    objects. Sync programs consume the same async fixtures in their worker.
    """
    import asyncio
    import inspect
    import json
    from contextlib import contextmanager
    from dataclasses import replace
    from unittest import mock

    from dspy._vendor.lm15.sse import SSEEvent
    from dspy.lm15 import OpenAIChatLM, StreamEndEvent, StreamStartEvent

    @contextmanager
    def patched():
        options = dict(kwargs)
        factory = options.pop("new_callable", mock.AsyncMock)
        recorded = factory(**options)

        async def astream(request):
            source = recorded()
            if inspect.isawaitable(source):
                source = await source
            yield StreamStartEvent(model=request.model)
            end = StreamEndEvent(finish_reason="stop")
            parser = OpenAIChatLM(api_key="test")
            async for chunk in source:
                events = parser.parse_stream_events(request, SSEEvent(event=None, data=json.dumps(chunk))) if isinstance(chunk, dict) else [chunk]
                for event in events:
                    if isinstance(event, StreamEndEvent):
                        end = replace(end, finish_reason=event.finish_reason or end.finish_reason,
                                      usage=event.usage or end.usage, provider_data=event.provider_data or end.provider_data)
                    else:
                        yield event
            yield end

        def stream(request):
            async def collect():
                return [event async for event in astream(request)]
            return iter(asyncio.run(collect()))

        with mock.patch("dspy.clients.engines.LM15Engine.stream", side_effect=stream), \
             mock.patch("dspy.clients.engines.AsyncLM15Engine.stream", side_effect=astream):
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
