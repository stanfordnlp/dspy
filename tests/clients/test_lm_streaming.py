"""Tests for the typed LM streaming contract: `stream()`, `forward_stream`, and SSE."""

from types import SimpleNamespace

import anyio
import pytest

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.clients.cache import Cache
from dspy.clients.openai_compat_lm import _OpenAICompatLM
from dspy.clients.openai_format import ChatCompletionChunkAssembler
from dspy.core.types import (
    LMResponse,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    response_to_stream_events,
)
from dspy.utils.exceptions import LMServerError

MODEL = "test-model"


class _TypedBufferedLM(BaseLM):
    """Typed LM without native streaming; exercises the buffered fallback."""

    forward_contract = "typed_lm"

    def __init__(self):
        super().__init__(model=MODEL, cache=False)
        self.calls = 0

    def forward(self, request):
        self.calls += 1
        return LMResponse.from_text("typed hi", model=MODEL, usage={"total_tokens": 3})

    async def aforward(self, request):
        return self.forward(request)


class _LegacyBufferedLM(BaseLM):
    """Legacy-contract LM; exercises the buffered fallback on the legacy path."""

    forward_contract = "legacy"

    def __init__(self):
        super().__init__(model=MODEL, cache=False)

    def forward(self, prompt=None, messages=None, **kwargs):
        return SimpleNamespace(
            id="chatcmpl-legacy",
            model=MODEL,
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(
                        role="assistant", content="legacy hi", reasoning_content=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )


class _NativeStreamLM(BaseLM):
    """Typed LM with a synchronous native stream; exercises the async bridge too."""

    forward_contract = "typed_lm"
    supports_streaming = True

    def __init__(self):
        super().__init__(model=MODEL, cache=False)

    def forward(self, request):
        raise AssertionError("stream tests must not fall back to forward()")

    def forward_stream(self, request):
        yield LMStreamStartEvent(model=MODEL)
        yield LMStreamDeltaEvent(part_index=0, delta=LMTextDelta(text="he"))
        yield LMStreamDeltaEvent(part_index=0, delta=LMTextDelta(text="llo"))
        yield LMStreamOutputEndEvent(finish_reason="stop")
        yield LMStreamEndEvent(usage={"total_tokens": 3})


def _event_types(events):
    return [event.type for event in events]


class TestBufferedFallback:
    def test_typed_lm_streams_buffered_response_as_events(self):
        lm = _TypedBufferedLM()
        stream = lm.stream("hi")
        events = list(stream)

        assert _event_types(events) == ["start", "delta", "output_end", "end"]
        assert events[1].delta.text == "typed hi"
        assert stream.result().text == "typed hi"
        assert lm.calls == 1

    def test_legacy_lm_streams_buffered_response_as_events(self):
        lm = _LegacyBufferedLM()
        stream = lm.stream("hi")
        events = list(stream)

        assert _event_types(events) == ["start", "delta", "output_end", "end"]
        assert stream.result().text == "legacy hi"

    def test_stream_records_history_once(self):
        lm = _TypedBufferedLM()
        list(lm.stream("hi"))
        assert len(lm.history) == 1
        entry = lm.history[-1]
        assert entry.response.text == "typed hi"

    def test_result_before_completion_raises(self):
        lm = _TypedBufferedLM()
        stream = lm.stream("hi")
        with pytest.raises(RuntimeError):
            stream.result()

    def test_astream_buffered_fallback(self):
        lm = _TypedBufferedLM()

        async def run():
            stream = lm.astream("hi")
            events = [event async for event in stream]
            return events, stream.result()

        events, response = anyio.run(run)
        assert _event_types(events) == ["start", "delta", "output_end", "end"]
        assert response.text == "typed hi"
        assert len(lm.history) == 1


class TestNativeStream:
    def test_stream_yields_native_events_and_result(self):
        lm = _NativeStreamLM()
        stream = lm.stream("hi")
        events = list(stream)

        assert _event_types(events) == ["start", "delta", "delta", "output_end", "end"]
        response = stream.result()
        assert response.text == "hello"
        assert response.usage.total_tokens == 3
        assert len(lm.history) == 1

    def test_astream_bridges_sync_forward_stream(self):
        lm = _NativeStreamLM()

        async def run():
            stream = lm.astream("hi")
            events = [event async for event in stream]
            return events, stream.result()

        events, response = anyio.run(run)
        assert _event_types(events) == ["start", "delta", "delta", "output_end", "end"]
        assert response.text == "hello"

    def test_astream_bridge_propagates_errors(self):
        class _FailingStreamLM(_NativeStreamLM):
            def forward_stream(self, request):
                yield LMStreamStartEvent(model=MODEL)
                raise LMServerError("boom", model=MODEL, provider="test")

        lm = _FailingStreamLM()

        async def run():
            async for _ in lm.astream("hi"):
                pass

        with pytest.raises(LMServerError):
            anyio.run(run)


class TestResponseToStreamEvents:
    def test_round_trips_multi_output_response(self):
        response = LMResponse.model_validate(
            {
                "model": MODEL,
                "outputs": [
                    {"parts": [{"type": "text", "text": "a"}], "finish_reason": "stop"},
                    {"parts": [{"type": "text", "text": "b"}], "finish_reason": "length", "truncated": True},
                ],
                "usage": {"total_tokens": 2},
            }
        )
        events = response_to_stream_events(response)
        assert _event_types(events) == ["start", "delta", "output_end", "delta", "output_end", "end"]
        assert events[-1].response is response


class TestChunkAssembler:
    def test_assigns_part_indices_in_order_of_appearance(self):
        assembler = ChatCompletionChunkAssembler()
        events = assembler.events(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "think",
                            "content": "text",
                            "tool_calls": [
                                {"index": 0, "id": "call_1", "function": {"name": "f", "arguments": "{"}}
                            ],
                        },
                    }
                ]
            }
        )
        assert [(event.part_index, event.delta.type) for event in events] == [
            (0, "thinking_delta"),
            (1, "text_delta"),
            (2, "tool_call_delta"),
        ]

        events = assembler.events({"choices": [{"index": 0, "delta": {"content": "more"}, "finish_reason": "length"}]})
        assert events[0].part_index == 1
        assert events[1].type == "output_end"
        assert events[1].truncated is True

    def test_reads_vllm_reasoning_key_and_usage(self):
        assembler = ChatCompletionChunkAssembler()
        events = assembler.events({"choices": [{"index": 0, "delta": {"reasoning": "hm"}}]})
        assert events[0].delta.type == "thinking_delta"

        assembler.events({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})
        end = assembler.end_events()[0]
        assert end.usage.total_tokens == 7


@pytest.fixture
def isolated_cache(tmp_path):
    original = dspy.cache
    dspy.cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir=tmp_path,
        memory_max_entries=100,
    )
    yield dspy.cache
    dspy.cache = original


def _chunks():
    return [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
        {"choices": [{"index": 0, "delta": {"content": "hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    ]


class TestOpenAICompatLMStreaming:
    def _make_lm(self, base_url, **kwargs):
        kwargs.setdefault("cache", False)
        return _OpenAICompatLM(model=MODEL, base_url=base_url, num_retries=0, **kwargs)

    def test_streams_sse_into_normalized_events(self, openai_compat_server):
        base_url, state = openai_compat_server
        state.reply_stream(_chunks())
        lm = self._make_lm(base_url)

        stream = lm.stream("hi")
        events = list(stream)
        response = stream.result()

        assert _event_types(events) == ["start", "delta", "delta", "delta", "output_end", "end"]
        assert response.text == "hello"
        assert response.outputs[0].parts[0].type == "thinking"
        assert response.outputs[0].finish_reason == "stop"
        assert response.usage.total_tokens == 5

        sent = state.requests[-1]["body"]
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}
        assert len(lm.history) == 1

    def test_stream_error_status_raises_typed_error(self, openai_compat_server):
        base_url, state = openai_compat_server
        state.reply(500, {"error": {"message": "kaput"}})
        lm = self._make_lm(base_url)

        with pytest.raises(LMServerError):
            list(lm.stream("hi"))

    def test_stream_populates_cache_and_replays_from_it(self, openai_compat_server, isolated_cache):
        base_url, state = openai_compat_server
        state.reply_stream(_chunks())
        lm = self._make_lm(base_url, cache=True)

        first = lm.stream("hi")
        list(first)
        replay = lm.stream("hi")
        events = list(replay)

        assert len(state.requests) == 1
        assert replay.result().text == first.result().text == "hello"
        assert events[0].type == "start" and events[-1].type == "end"

        # The buffered path shares the same cache entry: no new HTTP request.
        buffered = lm.forward(replay.request)
        assert len(state.requests) == 1
        assert buffered.text == "hello"

    def test_astream_streams_incrementally_via_thread_bridge(self, openai_compat_server):
        base_url, state = openai_compat_server
        state.reply_stream(_chunks())
        lm = self._make_lm(base_url)

        async def run():
            stream = lm.astream("hi")
            events = [event async for event in stream]
            return events, stream.result()

        events, response = anyio.run(run)
        assert _event_types(events) == ["start", "delta", "delta", "delta", "output_end", "end"]
        assert response.text == "hello"
