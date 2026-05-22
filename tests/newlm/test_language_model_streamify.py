import json

import pytest

import dspy


class AdapterStreamingLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="test/adapter-streaming", cache=False)
        self.stream_calls = 0
        self.forward_calls = 0

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_calls += 1
        return dspy.LMResponse.from_text(
            "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
            model=request.model,
        )

    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return self.forward(request)

    def forward_stream(self, request: dspy.LMRequest):
        self.stream_calls += 1
        yield dspy.lm.LMStreamStartEvent(model=request.model)
        for text in ["[[ ## answer ## ]]\n", "Par", "is", "\n\n[[ ## completed ## ]]"]:
            yield dspy.lm.LMStreamDeltaEvent(
                output_index=0,
                part_index=0,
                delta=dspy.lm.LMTextDelta(text=text),
            )
        yield dspy.lm.LMStreamOutputEndEvent(output_index=0, finish_reason="stop")
        yield dspy.lm.LMStreamEndEvent()

    async def aforward_stream(self, request: dspy.LMRequest):
        for event in self.forward_stream(request):
            yield event


@pytest.mark.asyncio
async def test_streamify_stream_listeners_work_with_language_model():
    lm = AdapterStreamingLM()
    predict = dspy.Predict("question -> answer")
    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    stream_predict = dspy.streamify(
        predict,
        stream_listeners=[listener],
        include_final_prediction_in_output_stream=False,
    )

    chunks = []
    final = None
    with dspy.context(lm=lm):
        async for value in stream_predict(question="Capital of France?"):
            if isinstance(value, dspy.Prediction):
                final = value
            else:
                chunks.append(value)

    assert lm.stream_calls == 1
    assert lm.forward_calls == 0
    assert "".join(chunk.chunk for chunk in chunks) == "Paris"
    assert chunks[-1].is_last_chunk is True
    assert final is None


@pytest.mark.asyncio
async def test_streamify_with_listeners_can_also_yield_raw_language_model_events():
    lm = AdapterStreamingLM()
    predict = dspy.Predict("question -> answer")
    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    stream_predict = dspy.streamify(
        predict,
        stream_listeners=[listener],
        include_lm_events=True,
    )

    field_chunks = []
    events = []
    final = None
    with dspy.context(lm=lm):
        async for value in stream_predict(question="Capital of France?"):
            if isinstance(value, dspy.Prediction):
                final = value
            elif isinstance(value, dspy.lm.LMStreamEvent):
                events.append(value)
            else:
                field_chunks.append(value)

    assert "".join(chunk.chunk for chunk in field_chunks) == "Paris"
    assert any(isinstance(event, dspy.lm.LMStreamStartEvent) for event in events)
    assert any(isinstance(event, dspy.lm.LMStreamEndEvent) for event in events)
    assert "".join(
        event.delta.text
        for event in events
        if isinstance(event, dspy.lm.LMStreamDeltaEvent) and isinstance(event.delta, dspy.lm.LMTextDelta)
    ) == "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
    assert final.answer == "Paris"


@pytest.mark.asyncio
async def test_streamify_without_listeners_yields_language_model_chunks_and_final_prediction():
    lm = AdapterStreamingLM()
    predict = dspy.Predict("question -> answer")
    stream_predict = dspy.streamify(predict)

    chunks = []
    final = None
    with dspy.context(lm=lm):
        async for value in stream_predict(question="Capital of France?"):
            if isinstance(value, dspy.Prediction):
                final = value
            else:
                chunks.append(value)

    assert lm.stream_calls == 1
    assert all(isinstance(chunk, dspy.lm.LMStreamEvent) for chunk in chunks)
    assert "".join(
        event.delta.text
        for event in chunks
        if isinstance(event, dspy.lm.LMStreamDeltaEvent) and isinstance(event.delta, dspy.lm.LMTextDelta)
    ) == "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
    assert final.answer == "Paris"


@pytest.mark.asyncio
async def test_streaming_response_serializes_normalized_lm_stream_events():
    async def streamer():
        yield dspy.lm.LMStreamStartEvent(model="test/model")
        yield dspy.lm.LMStreamDeltaEvent(part_index=0, delta=dspy.lm.LMTextDelta(text="hello"))

    chunks = [chunk async for chunk in dspy.streaming.streaming_response(streamer())]

    assert json.loads(chunks[0].removeprefix("data: ")) == {"event": {"type": "start", "model": "test/model"}}
    assert json.loads(chunks[1].removeprefix("data: "))["event"]["delta"] == {"type": "text_delta", "text": "hello"}
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_streaming_response_serializes_stream_responses_and_status_messages():
    async def streamer():
        yield dspy.streaming.StatusMessage("working")
        yield dspy.streaming.StreamResponse("predict", "answer", "Par", is_last_chunk=False)
        yield dspy.streaming.StreamResponse("predict", "answer", "is", is_last_chunk=True)

    chunks = [chunk async for chunk in dspy.streaming.streaming_response(streamer())]

    assert json.loads(chunks[0].removeprefix("data: ")) == {"status": {"message": "working"}}
    assert json.loads(chunks[1].removeprefix("data: ")) == {
        "stream_response": {
            "predict_name": "predict",
            "signature_field_name": "answer",
            "chunk": "Par",
            "is_last_chunk": False,
        }
    }
    assert json.loads(chunks[2].removeprefix("data: "))["stream_response"]["is_last_chunk"] is True
    assert chunks[-1] == "data: [DONE]\n\n"
