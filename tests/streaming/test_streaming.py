import asyncio
import time
from dataclasses import dataclass
from unittest import mock
from unittest.mock import AsyncMock

import pydantic
import pytest
from asyncer import syncify

import dspy
from dspy.adapters.types import Type
from dspy.clients._request_utils import StreamChunk
from dspy.experimental import Citations, Document
from dspy.streaming import StatusMessage, StatusMessageProvider, StreamResponse, streaming_response


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class MockStreamWrapper:
    """Mimics the async-iterator wrapper returned by ``backend.astream_complete``.

    Yields ``StreamChunk`` objects, then sets ``.assembled`` to a fake
    ``ChatCompletion`` once exhaustion completes.
    """

    def __init__(self, chunks: list[StreamChunk], model: str = "mock-model"):
        self._chunks = list(chunks)
        self._idx = 0
        self._model = model
        self.assembled = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamChunk:
        if self._idx >= len(self._chunks):
            self.assembled = self._build_response()
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    def _build_response(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types import CompletionUsage

        text = "".join(c.content or "" for c in self._chunks)
        reasoning = "".join(c.reasoning_content or "" for c in self._chunks)
        msg_kwargs = {"role": "assistant", "content": text or None}
        if reasoning:
            msg_kwargs["reasoning_content"] = reasoning
        return ChatCompletion(
            id="mock-stream",
            choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(**msg_kwargs))],
            created=0,
            model=self._model,
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


def _sc(content=None, reasoning_content=None, provider_specific_fields=None):
    """Shorthand to create a StreamChunk."""
    return StreamChunk(
        content=content,
        reasoning_content=reasoning_content,
        provider_specific_fields=provider_specific_fields,
    )


def _mock_astream(chunks, model="mock-model"):
    """Return an ``AsyncMock`` that returns a ``MockStreamWrapper``."""
    wrapper = MockStreamWrapper(chunks, model=model)

    async def _factory(request, num_retries):
        return wrapper

    return _factory


def _mock_astream_factory(chunk_lists, model="mock-model"):
    """Return a side-effect function that returns successive ``MockStreamWrapper`` instances."""
    wrappers = [MockStreamWrapper(cl, model=model) for cl in chunk_lists]
    idx = {"i": 0}

    async def _factory(request, num_retries):
        w = wrappers[idx["i"]]
        idx["i"] += 1
        return w

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_streaming_cache_hit_skips_streaming():
    """Second call with cache=True should return the cached response without streaming."""
    import uuid

    my_program = dspy.Predict("question->answer")
    program = dspy.streamify(
        my_program,
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    )

    chunks = [
        _sc("[[ ## answer ## ]]\n"),
        _sc("Hello!"),
        _sc("\n\n[[ ## completed ## ]]"),
    ]

    call_count = {"n": 0}

    async def counting_astream(request, num_retries):
        call_count["n"] += 1
        return MockStreamWrapper(chunks)

    # Use a unique question to avoid hitting disk cache from previous runs.
    unique_q = f"cache_test_{uuid.uuid4().hex[:8]}"

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=counting_astream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=True), adapter=dspy.ChatAdapter()):
            # First call — should stream
            output1 = program(question=unique_q)
            chunks1 = [v async for v in output1]
            pred1 = [c for c in chunks1 if isinstance(c, dspy.Prediction)]
            assert len(pred1) == 1
            assert pred1[0].answer == "Hello!"
            assert call_count["n"] == 1

            # Second call (same input) — should hit cache, no streaming
            output2 = program(question=unique_q)
            chunks2 = [v async for v in output2]
            assert len(chunks2) == 1  # only the Prediction
            assert isinstance(chunks2[0], dspy.Prediction)
            assert chunks2[0].answer == "Hello!"
            # astream_complete should NOT have been called again
            assert call_count["n"] == 1


@pytest.mark.anyio
async def test_streamify_yields_expected_response_chunks(litellm_test_server):
    api_base, _ = litellm_test_server
    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        cache=True,
    )
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):

        class TestSignature(dspy.Signature):
            input_text: str = dspy.InputField()
            output_text: str = dspy.OutputField()

        program = dspy.streamify(dspy.Predict(TestSignature))
        output_stream1 = program(input_text="Test")
        output_chunks1 = [chunk async for chunk in output_stream1]
        last_chunk1 = output_chunks1[-1]
        assert isinstance(last_chunk1, dspy.Prediction)
        assert last_chunk1.output_text == "Hello!"

        output_stream2 = program(input_text="Test")
        output_chunks2 = [chunk async for chunk in output_stream2]
        # Since the input is cached, only one chunk should be
        # yielded containing the prediction
        assert len(output_chunks2) == 1
        last_chunk2 = output_chunks2[-1]
        assert isinstance(last_chunk2, dspy.Prediction)
        assert last_chunk2.output_text == "Hello!"


@pytest.mark.anyio
async def test_streaming_response_yields_expected_response_chunks(litellm_test_server):
    api_base, _ = litellm_test_server
    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        cache=False,
    )
    with dspy.context(lm=lm):

        class TestSignature(dspy.Signature):
            input_text: str = dspy.InputField()
            output_text: str = dspy.OutputField()

        program = dspy.streamify(dspy.Predict(TestSignature))
        output_stream_from_program = streaming_response(program(input_text="Test"))
        output_stream_for_server_response = streaming_response(output_stream_from_program)
        output_chunks = [chunk async for chunk in output_stream_for_server_response]
        assert all(chunk.startswith("data: ") for chunk in output_chunks)
        assert 'data: {"prediction":{"output_text":"Hello!"}}\n\n' in output_chunks
        assert output_chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_default_status_streaming():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.generate_question = dspy.Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = dspy.Predict("question->answer")

        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    lm = dspy.utils.DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with dspy.context(lm=lm):
        program = dspy.streamify(MyProgram())
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)

    assert len(status_messages) == 2
    assert status_messages[0].message == "Calling tool generate_question..."
    assert status_messages[1].message == "Tool calling finished! Querying the LLM with tool calling results..."


@pytest.mark.anyio
async def test_custom_status_streaming():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.generate_question = dspy.Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = dspy.Predict("question->answer")

        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    class MyStatusMessageProvider(StatusMessageProvider):
        def tool_start_status_message(self, instance, inputs):
            return "Tool starting!"

        def tool_end_status_message(self, outputs):
            return "Tool finished!"

        def module_start_status_message(self, instance, inputs):
            if isinstance(instance, dspy.Predict):
                return "Predict starting!"

    lm = dspy.utils.DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with dspy.context(lm=lm):
        program = dspy.streamify(MyProgram(), status_message_provider=MyStatusMessageProvider())
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)

        assert len(status_messages) == 3
        assert status_messages[0].message == "Tool starting!"
        assert status_messages[1].message == "Tool finished!"
        assert status_messages[2].message == "Predict starting!"


@pytest.mark.anyio
async def test_concurrent_status_message_providers():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.generate_question = dspy.Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = dspy.Predict("question->answer")

        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    class MyStatusMessageProvider1(StatusMessageProvider):
        def tool_start_status_message(self, instance, inputs):
            return "Provider1: Tool starting!"

        def tool_end_status_message(self, outputs):
            return "Provider1: Tool finished!"

        def module_start_status_message(self, instance, inputs):
            if isinstance(instance, dspy.Predict):
                return "Provider1: Predict starting!"

    class MyStatusMessageProvider2(StatusMessageProvider):
        def tool_start_status_message(self, instance, inputs):
            return "Provider2: Tool starting!"

        def tool_end_status_message(self, outputs):
            return "Provider2: Tool finished!"

        def module_start_status_message(self, instance, inputs):
            if isinstance(instance, dspy.Predict):
                return "Provider2: Predict starting!"

    # Store the original callbacks to verify they're not modified
    original_callbacks = list(dspy.settings.callbacks)

    lm = dspy.utils.DummyLM([{"answer": "red"}, {"answer": "blue"}, {"answer": "green"}, {"answer": "yellow"}])

    # Results storage for each thread
    results = {}

    async def run_with_provider1():
        with dspy.context(lm=lm):
            program = dspy.streamify(MyProgram(), status_message_provider=MyStatusMessageProvider1())
            output = program("sky")

            status_messages = []
            async for value in output:
                if isinstance(value, StatusMessage):
                    status_messages.append(value.message)

            results["provider1"] = status_messages

    async def run_with_provider2():
        with dspy.context(lm=lm):
            program = dspy.streamify(MyProgram(), status_message_provider=MyStatusMessageProvider2())
            output = program("ocean")

            status_messages = []
            async for value in output:
                if isinstance(value, StatusMessage):
                    status_messages.append(value.message)

            results["provider2"] = status_messages

    # Run both tasks concurrently
    await asyncio.gather(run_with_provider1(), run_with_provider2())

    # Verify provider1 got its expected messages
    assert len(results["provider1"]) == 3
    assert results["provider1"][0] == "Provider1: Tool starting!"
    assert results["provider1"][1] == "Provider1: Tool finished!"
    assert results["provider1"][2] == "Provider1: Predict starting!"

    # Verify provider2 got its expected messages
    assert len(results["provider2"]) == 3
    assert results["provider2"][0] == "Provider2: Tool starting!"
    assert results["provider2"][1] == "Provider2: Tool finished!"
    assert results["provider2"][2] == "Provider2: Predict starting!"

    # Verify that the global callbacks were not modified
    assert dspy.settings.callbacks == original_callbacks


@pytest.mark.llm_call
@pytest.mark.anyio
async def test_stream_listener_chat_adapter(lm_for_test):
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->judgement")

        def __call__(self, x: str, **kwargs):
            answer = self.predict1(question=x, **kwargs)
            judgement = self.predict2(question=x, answer=answer, **kwargs)
            return judgement

    my_program = MyProgram()
    program = dspy.streamify(
        my_program,
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer"),
            dspy.streaming.StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
    )
    # Turn off the cache to ensure the stream is produced.
    with dspy.context(lm=dspy.LM(lm_for_test, cache=False, temperature=0.0)):
        output = program(x="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, dspy.streaming.StreamResponse):
                all_chunks.append(value)

    assert all_chunks[0].predict_name == "predict1"
    assert all_chunks[0].signature_field_name == "answer"
    # The last chunk can be from either predictor because sometimes small LMs miss the `[[ ## completed ## ]]` marker,
    # which results in an extra chunk that flushes out the buffer.
    assert all_chunks[-2].predict_name == "predict2"
    assert all_chunks[-2].signature_field_name == "judgement"


@pytest.mark.anyio
async def test_default_status_streaming_in_async_program():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.generate_question = dspy.Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = dspy.Predict("question->answer")

        async def acall(self, x: str):
            question = await self.generate_question.acall(x=x)
            return await self.predict.acall(question=question)

    lm = dspy.utils.DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with dspy.context(lm=lm):
        program = dspy.streamify(MyProgram(), is_async_program=True)
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)

    assert len(status_messages) == 2
    assert status_messages[0].message == "Calling tool generate_question..."
    assert status_messages[1].message == "Tool calling finished! Querying the LLM with tool calling results..."


@pytest.mark.llm_call
@pytest.mark.anyio
async def test_stream_listener_json_adapter(lm_for_test):
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->judgement")

        def __call__(self, x: str, **kwargs):
            answer = self.predict1(question=x, **kwargs)
            judgement = self.predict2(question=x, answer=answer, **kwargs)
            return judgement

    my_program = MyProgram()
    program = dspy.streamify(
        my_program,
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer"),
            dspy.streaming.StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
    )
    # Turn off the cache to ensure the stream is produced.
    with dspy.context(lm=dspy.LM(lm_for_test, cache=False, temperature=0.0), adapter=dspy.JSONAdapter()):
        output = program(x="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, dspy.streaming.StreamResponse):
                all_chunks.append(value)

    assert all_chunks[0].predict_name == "predict1"
    assert all_chunks[0].signature_field_name == "answer"
    assert all_chunks[0].is_last_chunk is False

    assert all_chunks[-1].predict_name == "predict2"
    assert all_chunks[-1].signature_field_name == "judgement"


@pytest.mark.anyio
async def test_streaming_handles_space_correctly():
    my_program = dspy.Predict("question->answer")
    program = dspy.streamify(
        my_program, stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")]
    )

    chunks = [
        _sc("[[ ## answer ## ]]\n"),
        _sc("How "),
        _sc("are "),
        _sc("you "),
        _sc("doing?"),
        _sc("\n\n[[ ## completed ## ]]"),
    ]

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="What is the capital of France?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    assert "".join([chunk.chunk for chunk in all_chunks]) == "How are you doing?"


@pytest.mark.llm_call
def test_sync_streaming(lm_for_test):
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->judgement")

        def __call__(self, x: str, **kwargs):
            answer = self.predict1(question=x, **kwargs)
            judgement = self.predict2(question=x, answer=answer, **kwargs)
            return judgement

    my_program = MyProgram()
    program = dspy.streamify(
        my_program,
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer"),
            dspy.streaming.StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
        async_streaming=False,
    )
    # Turn off the cache to ensure the stream is produced.
    with dspy.context(lm=dspy.LM(lm_for_test, cache=False, temperature=0.0)):
        output = program(x="why did a chicken cross the kitchen?")
        all_chunks = []
        for value in output:
            if isinstance(value, dspy.streaming.StreamResponse):
                all_chunks.append(value)

    assert all_chunks[0].predict_name == "predict1"
    assert all_chunks[0].signature_field_name == "answer"
    assert all_chunks[0].is_last_chunk is False
    # The last chunk can be from either predictor because sometimes small LMs miss the `[[ ## completed ## ]]` marker,
    # which results in an extra chunk that flushes out the buffer.
    assert all_chunks[-2].predict_name == "predict2"
    assert all_chunks[-2].signature_field_name == "judgement"


def test_sync_status_streaming():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.generate_question = dspy.Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = dspy.Predict("question->answer")

        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    lm = dspy.utils.DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with dspy.context(lm=lm):
        program = dspy.streamify(MyProgram())
        output = program("sky")
        sync_output = dspy.streaming.apply_sync_streaming(output)
        status_messages = []
        for value in sync_output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)

    assert len(status_messages) == 2
    assert status_messages[0].message == "Calling tool generate_question..."
    assert status_messages[1].message == "Tool calling finished! Querying the LLM with tool calling results..."


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_chat_adapter():
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->judgement")

        def forward(self, question, **kwargs):
            answer = self.predict1(question=question, **kwargs).answer
            judgement = self.predict2(question=question, answer=answer, **kwargs)
            return judgement

    stream_1_chunks = [
        _sc("[["), _sc(" ##"), _sc(" answer"), _sc(" ##"), _sc(" ]]\n\n"),
        _sc("To"), _sc(" get"), _sc(" to"), _sc(" the"), _sc(" other"),
        _sc(" side"), _sc(" of"), _sc(" the"), _sc(" dinner"), _sc(" plate"),
        _sc("!\n\n[[ ##"), _sc(" completed"), _sc(" ##"), _sc(" ]]"),
    ]

    stream_2_chunks = [
        _sc("[[ ##"), _sc(" judgement"), _sc(" ##"), _sc(" ]]\n\n"),
        _sc("The"), _sc(" answer"), _sc(" is"), _sc(" humorous"),
        _sc(" and"), _sc(" plays"), _sc(" on"), _sc(" the"),
        _sc(" classic"), _sc(" joke"), _sc(" format"),
        _sc(".\n\n[[ ##"), _sc(" completed"), _sc(" ##"), _sc(" ]]"),
    ]

    with mock.patch(
        "dspy.clients._openai.astream_complete",
        side_effect=_mock_astream_factory([stream_1_chunks, stream_2_chunks]),
    ):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

        assert all_chunks[0].predict_name == "predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == "To"
        assert all_chunks[1].chunk == " get"
        assert all_chunks[2].chunk == " to"
        assert all_chunks[3].chunk == " the"
        assert all_chunks[4].chunk == " other"
        assert all_chunks[5].chunk == " side"
        assert all_chunks[6].chunk == " of"
        assert all_chunks[7].chunk == " the"
        assert all_chunks[8].chunk == " dinner"
        assert all_chunks[9].chunk == " plate"
        assert all_chunks[10].chunk == "!"
        assert all_chunks[10].is_last_chunk is True

        assert all_chunks[11].predict_name == "predict2"
        assert all_chunks[11].signature_field_name == "judgement"
        assert all_chunks[11].chunk == "The"
        assert all_chunks[12].chunk == " answer"
        assert all_chunks[13].chunk == " is"
        assert all_chunks[14].chunk == " humorous"
        assert all_chunks[15].chunk == " and"
        assert all_chunks[16].chunk == " plays"
        assert all_chunks[17].chunk == " on"
        assert all_chunks[18].chunk == " the"
        assert all_chunks[19].chunk == " classic"
        assert all_chunks[20].chunk == " joke"
        assert all_chunks[21].chunk == " format"
        assert all_chunks[22].chunk == "."
        assert all_chunks[22].is_last_chunk is True


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_json_adapter():
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question,answer->judgement")

        def forward(self, question, **kwargs):
            answer = self.predict1(question=question, **kwargs).answer
            judgement = self.predict2(question=question, answer=answer, **kwargs)
            return judgement

    stream_1_chunks = [
        _sc('{"'), _sc("answer"), _sc('":'), _sc('"To'), _sc(" get"),
        _sc(" to"), _sc(" the"), _sc(" other"), _sc(" side"), _sc(" of"),
        _sc(" the"), _sc(" frying"), _sc(" pan"), _sc('!"'), _sc("}\n"),
        _sc("None"), _sc("None"), _sc("None"),
    ]

    stream_2_chunks = [
        _sc('{"'), _sc("jud"), _sc("gement"), _sc('":'), _sc('"The'),
        _sc(" answer"), _sc(" is"), _sc(" humorous"), _sc(" and"),
        _sc(" plays"), _sc(" on"), _sc(" the"), _sc(" very"), _sc(" funny"),
        _sc(" and"), _sc(" classic"), _sc(" joke"), _sc(" format"),
        _sc('."'), _sc("}"), _sc("None"), _sc("None"), _sc("None"),
    ]

    with mock.patch(
        "dspy.clients._openai.astream_complete",
        side_effect=_mock_astream_factory([stream_1_chunks, stream_2_chunks]),
    ):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

        assert all_chunks[0].predict_name == "predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == '"To'
        assert all_chunks[1].chunk == " get"
        assert all_chunks[2].chunk == " to"
        assert all_chunks[3].chunk == " the"
        assert all_chunks[4].chunk == " other"
        assert all_chunks[5].chunk == " side"
        assert all_chunks[6].chunk == " of"
        assert all_chunks[7].chunk == " the"
        assert all_chunks[8].chunk == " frying"
        assert all_chunks[9].chunk == " pan"
        assert all_chunks[10].chunk == '!"'
        assert all_chunks[10].is_last_chunk is True

        assert all_chunks[11].predict_name == "predict2"
        assert all_chunks[11].signature_field_name == "judgement"
        assert all_chunks[11].chunk == '"The'
        assert all_chunks[12].chunk == " answer"
        assert all_chunks[13].chunk == " is"
        assert all_chunks[14].chunk == " humorous"
        assert all_chunks[15].chunk == " and"
        assert all_chunks[16].chunk == " plays"
        assert all_chunks[17].chunk == " on"
        assert all_chunks[18].chunk == " the"
        assert all_chunks[19].chunk == " very"
        assert all_chunks[20].chunk == " funny"
        assert all_chunks[21].chunk == " and"
        assert all_chunks[22].chunk == " classic"
        assert all_chunks[23].chunk == " joke"
        assert all_chunks[24].chunk == " format"
        assert all_chunks[25].chunk == '."'
        assert all_chunks[25].is_last_chunk is True


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_chat_adapter_untokenized_stream():
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question,answer->judgement")

        def forward(self, question, **kwargs):
            answer = self.predict1(question=question, **kwargs).answer
            judgement = self.predict2(question=question, answer=answer, **kwargs)
            return judgement

    stream_1_chunks = [
        _sc("[[ ##"), _sc(" answer ## ]]"),
        _sc("To get to the other side."),
        _sc("\n\n[[ ## completed ## ]]"),
    ]

    stream_2_chunks = [
        _sc("[[ ## judgement ## ]]\n\n"),
        _sc(
            "The answer provides the standard punchline for this classic joke format, adapted to the "
            "specific location mentioned in the question. It is the expected and appropriate response."
        ),
        _sc("\n\n[[ ## completed ## ]]"),
        _sc("}\n"),
    ]

    with mock.patch(
        "dspy.clients._google.astream_complete",
        side_effect=_mock_astream_factory([stream_1_chunks, stream_2_chunks], model="gemini-2.5-flash"),
    ):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("gemini/gemini-2.5-flash", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

        assert all_chunks[0].predict_name == "predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == "To get to the other side."
        assert all_chunks[1].is_last_chunk is True

        assert all_chunks[2].predict_name == "predict2"
        assert all_chunks[2].signature_field_name == "judgement"
        assert all_chunks[2].chunk == (
            "The answer provides the standard punchline for this classic joke format, adapted to the specific location "
            "mentioned in the question. It is the expected and appropriate response."
        )


@pytest.mark.anyio
async def test_stream_listener_missing_completion_marker_chat_adapter():
    """Test that streaming works correctly when LLM response omits a final completion marker."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    chunks = [
        _sc("[[ ##"), _sc(" answer"), _sc(" ## ]]\n\n"),
        _sc("This"), _sc(" is"), _sc(" a"), _sc(" test"), _sc(" response"),
        _sc(" with"), _sc(" many"), _sc(" tokens"), _sc(" to"),
        _sc(" ensure"), _sc(" buffering"), _sc(" works"), _sc(" correctly"),
        _sc("."),
        # NO COMPLETION MARKER
    ]

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Test question")
            all_chunks = []
            final_prediction = None
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)
                elif isinstance(value, dspy.Prediction):
                    final_prediction = value

    full_content = "".join([chunk.chunk for chunk in all_chunks])
    expected_content = "This is a test response with many tokens to ensure buffering works correctly."
    assert full_content == expected_content
    assert final_prediction.answer == expected_content


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_json_adapter_untokenized_stream():
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question,answer->judgement")

        def forward(self, question, **kwargs):
            answer = self.predict1(question=question, **kwargs).answer
            judgement = self.predict2(question=question, answer=answer, **kwargs)
            return judgement

    stream_1_chunks = [
        _sc("{\n"),
        _sc('  "answer": "To get to'),
        _sc(' the other side... of the cutting board!"'),
        _sc("}\n"),
    ]

    stream_2_chunks = [
        _sc("{\n"),
        _sc('  "judgement": "The'),
        _sc(' answer provides a humorous and relevant punchline to the classic joke setup."'),
        _sc("}\n"),
    ]

    with mock.patch(
        "dspy.clients._google.astream_complete",
        side_effect=_mock_astream_factory([stream_1_chunks, stream_2_chunks], model="gemini-2.5-flash"),
    ):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("gemini/gemini-2.5-flash", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

        assert all_chunks[0].predict_name == "predict1"
        assert all_chunks[0].signature_field_name == "answer"

        assert all_chunks[0].chunk == '"To get to the other side... of the cutting board!"'

        assert all_chunks[1].predict_name == "predict2"
        assert all_chunks[1].signature_field_name == "judgement"
        assert (
            all_chunks[1].chunk == '"The answer provides a humorous and relevant punchline to the classic joke setup."'
        )


@pytest.mark.anyio
async def test_status_message_non_blocking():
    def dummy_tool():
        time.sleep(1)
        return "dummy_tool_output"

    class MyProgram(dspy.Module):
        def forward(self, question, **kwargs):
            dspy.Tool(dummy_tool)()
            return dspy.Prediction(answer="dummy_tool_output")

    program = dspy.streamify(MyProgram(), status_message_provider=StatusMessageProvider())

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
        output = program(question="why did a chicken cross the kitchen?")
        timestamps = []
        async for value in output:
            if isinstance(value, dspy.streaming.StatusMessage):
                timestamps.append(time.time())

    # timestamps[0]: tool start message
    # timestamps[1]: tool end message
    assert timestamps[1] - timestamps[0] >= 1


@pytest.mark.anyio
async def test_status_message_non_blocking_async_program():
    async def dummy_tool():
        await asyncio.sleep(1)
        return "dummy_tool_output"

    class MyProgram(dspy.Module):
        async def aforward(self, question, **kwargs):
            await dspy.Tool(dummy_tool).acall()
            return dspy.Prediction(answer="dummy_tool_output")

    program = dspy.streamify(MyProgram(), status_message_provider=StatusMessageProvider(), is_async_program=True)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
        output = program(question="why did a chicken cross the kitchen?")
        timestamps = []
        async for value in output:
            if isinstance(value, dspy.streaming.StatusMessage):
                timestamps.append(time.time())

    assert timestamps[1] - timestamps[0] >= 1


@pytest.mark.anyio
async def test_stream_listener_allow_reuse():
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            self.predict(question=question, **kwargs)
            return self.predict(question=question, **kwargs)

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer", allow_reuse=True),
        ],
    )

    reuse_chunks = [
        _sc("[["), _sc(" ##"), _sc(" answer"), _sc(" ##"), _sc(" ]]\n\n"),
        _sc("To"), _sc(" get"), _sc(" to"), _sc(" the"), _sc(" other"),
        _sc(" side"), _sc("!\n\n[[ ##"), _sc(" completed"), _sc(" ##"), _sc(" ]]"),
    ]

    with mock.patch(
        "dspy.clients._openai.astream_complete",
        side_effect=_mock_astream_factory([reuse_chunks, reuse_chunks]),
    ):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    concat_message = "".join([chunk.chunk for chunk in all_chunks])
    # The listener functions twice.
    assert concat_message == "To get to the other side!To get to the other side!"


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_xml_adapter():
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question,answer->judgement")

        def forward(self, question, **kwargs):
            answer = self.predict1(question=question, **kwargs).answer
            judgement = self.predict2(question=question, answer=answer, **kwargs)
            return judgement

    stream_1_chunks = [
        _sc("<"), _sc("answer"), _sc(">"),
        _sc("To"), _sc(" get"), _sc(" to"), _sc(" the"), _sc(" other"), _sc(" side"), _sc("!"),
        _sc("<"), _sc("/answer"), _sc(">"),
    ]

    stream_2_chunks = [
        _sc("<"), _sc("judgement"), _sc(">"),
        _sc("The"), _sc(" answer"), _sc(" is"), _sc(" humorous"), _sc("."),
        _sc("<"), _sc("/judgement"), _sc(">"),
    ]

    with mock.patch(
        "dspy.clients._openai.astream_complete",
        side_effect=_mock_astream_factory([stream_1_chunks, stream_2_chunks]),
    ):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Verify answer chunks
    answer_chunks = [chunk for chunk in all_chunks if chunk.signature_field_name == "answer"]
    assert len(answer_chunks) > 0
    assert answer_chunks[0].predict_name == "predict1"
    assert "".join([chunk.chunk for chunk in answer_chunks]) == "To get to the other side!"

    # Verify judgement chunks
    judgement_chunks = [chunk for chunk in all_chunks if chunk.signature_field_name == "judgement"]
    assert len(judgement_chunks) > 0
    assert judgement_chunks[0].predict_name == "predict2"
    assert "".join([chunk.chunk for chunk in judgement_chunks]) == "The answer is humorous."


@pytest.mark.anyio
async def test_streaming_allows_custom_chunk_types():
    @dataclass
    class CustomChunk:
        text: str

    class MyProgram(dspy.Module):
        def forward(self, question, **kwargs):
            async def send_to_stream():
                chunk = CustomChunk(text="hello")
                await dspy.settings.send_stream.send(chunk)

            syncified_send_to_stream = syncify(send_to_stream)
            syncified_send_to_stream()
            return dspy.Prediction(answer="dummy output")

    program = dspy.streamify(MyProgram())

    output = program(question="why did a chicken cross the kitchen?")
    all_chunks = []
    async for value in output:
        all_chunks.append(value)

    assert isinstance(all_chunks[0], CustomChunk)
    assert isinstance(all_chunks[1], dspy.Prediction)


@pytest.mark.anyio
async def test_streaming_allows_custom_streamable_type():
    class CustomType(Type):
        message: str

        @classmethod
        def is_streamable(cls) -> bool:
            return True

        @classmethod
        def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
            return signature.delete(field_name)

        @classmethod
        def parse_stream_chunk(cls, chunk):
            return CustomType(message=chunk.choices[0].delta.content)

        @classmethod
        def parse_lm_response(cls, response: dict) -> "CustomType":
            return CustomType(message=response.split("\n\n")[0])

    class CustomSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: CustomType = dspy.OutputField()

    program = dspy.streamify(
        dspy.Predict(CustomSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer"),
        ],
    )

    chunks = [
        _sc("Hello"), _sc("World"), _sc("\n\n"),
        _sc("[[ ##"), _sc(" completed"), _sc(" ##"), _sc(" ]]"),
    ]

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter(native_response_types=[CustomType])
        ):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)
                elif isinstance(value, dspy.Prediction):
                    assert isinstance(value.answer, CustomType)
                    assert value.answer.message == "HelloWorld"

    assert all(isinstance(chunk.chunk, CustomType) for chunk in all_chunks)


@pytest.mark.anyio
async def test_streaming_with_citations():
    class AnswerWithSources(dspy.Signature):
        """Answer questions using provided documents with citations."""

        documents: list[Document] = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        citations: Citations = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(AnswerWithSources)

        def forward(self, documents, question, **kwargs):
            return self.predict(documents=documents, question=question, **kwargs)

    chunks = [
        _sc("[[ ##"), _sc(" answer"), _sc(" ## ]]\n\n"),
        _sc("A"), _sc("c"), _sc("c"), _sc("o"), _sc("r"), _sc("d"),
        _sc("i"), _sc("n"), _sc("g"), _sc(" to "), _sc("the references,"),
        StreamChunk(
            content="",
            provider_specific_fields={
                "citation": {
                    "type": "char_location",
                    "cited_text": "water boils at 100°C",
                    "document_index": 0,
                    "document_title": "Physics Facts",
                    "start_char_index": 0,
                    "end_char_index": 19,
                }
            },
        ),
        _sc(" water"), _sc(" boils"), _sc(" at"), _sc(" 100°C"),
        _sc(".\n\n[[ ##"), _sc(" completed"), _sc(" ## ]]"),
    ]

    with mock.patch(
        "dspy.clients._anthropic.astream_complete",
        side_effect=_mock_astream(chunks, model="claude-3-5-sonnet-20241022"),
    ):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="citations"),
            ],
        )

        # Create test documents
        docs = [Document(data="Water boils at 100°C at standard pressure.", title="Physics Facts")]

        with dspy.context(
            lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022", cache=False),
            adapter=dspy.ChatAdapter(native_response_types=[Citations]),
        ):
            output = program(documents=docs, question="What temperature does water boil?")
            citation_chunks = []
            answer_chunks = []
            final_prediction = None
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse) and value.signature_field_name == "citations":
                    citation_chunks.append(value)
                elif isinstance(value, dspy.streaming.StreamResponse) and value.signature_field_name == "answer":
                    answer_chunks.append(value.chunk)
                elif isinstance(value, dspy.Prediction):
                    final_prediction = value

            # Test that we received citation chunks from streaming
            assert len(citation_chunks) > 0
            citation_chunk = citation_chunks[0]
            assert isinstance(citation_chunk.chunk, Citations)
            assert len(citation_chunk.chunk) == 1
            assert citation_chunk.chunk[0].cited_text == "water boils at 100°C"
            assert citation_chunk.chunk[0].document_title == "Physics Facts"

            # Verify the answer chunks are correct
            assert "".join(answer_chunks) == "According to the references, water boils at 100°C."

            # Test that prediction contains the expected fields
            assert final_prediction is not None
            assert hasattr(final_prediction, "answer")
            assert hasattr(final_prediction, "citations")


# Test Pydantic Models
class SimpleResponse(pydantic.BaseModel):
    message: str
    status: str


class NestedResponse(pydantic.BaseModel):
    title: str
    content: dict
    metadata: SimpleResponse


class ComplexResponse(pydantic.BaseModel):
    items: list[str]
    settings: dict[str, str]
    active: bool


@pytest.mark.anyio
async def test_chat_adapter_simple_pydantic_streaming():
    """Test ChatAdapter streaming with a simple pydantic model."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    chunks = [
        _sc("[[ ##"), _sc(" response"), _sc(" ## ]]\n\n"),
        _sc('{"message": "Hello'), _sc(' world!"'),
        _sc(', "status":'), _sc(' "success"}'),
        _sc("\n\n[[ ##"), _sc(" completed"), _sc(" ## ]]"),
    ]

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Say hello")
            chunks_out = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks_out.append(value)

    assert len(chunks_out) > 0
    assert chunks_out[0].signature_field_name == "response"
    full_content = "".join(chunk.chunk for chunk in chunks_out)
    assert "Hello world!" in full_content
    assert "success" in full_content


@pytest.mark.anyio
async def test_chat_adapter_with_generic_type_annotation():
    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: list[str] | int = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    chunks = [
        _sc("[[ ##"), _sc(" response"), _sc(" ## ]]\n\n"),
        _sc("1"),
        _sc("\n\n[[ ##"), _sc(" completed"), _sc(" ## ]]"),
    ]

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Say hello")
            chunks_out = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks_out.append(value)

    assert len(chunks_out) > 0
    assert chunks_out[0].signature_field_name == "response"
    full_content = "".join(chunk.chunk for chunk in chunks_out)
    assert "1" in full_content


@pytest.mark.anyio
async def test_chat_adapter_nested_pydantic_streaming():
    """Test ChatAdapter streaming with nested pydantic model."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: NestedResponse = dspy.OutputField()

    chunks = [
        _sc("[[ ## response ## ]]\n\n"),
        _sc('{"title": "Test"'),
        _sc(', "content": {"key": "value"}'),
        _sc(', "metadata": {"message": "nested"'),
        _sc(', "status": "ok"}}'),
        _sc("\n\n[[ ## completed ## ]]"),
    ]

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Generate nested response")
            chunks_out = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks_out.append(value)

    assert len(chunks_out) > 0
    full_content = "".join(chunk.chunk for chunk in chunks_out)
    assert "nested" in full_content
    assert "Test" in full_content


@pytest.mark.anyio
async def test_chat_adapter_mixed_fields_streaming():
    """Test ChatAdapter streaming with both pydantic and string fields."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        summary: str = dspy.OutputField()
        details: SimpleResponse = dspy.OutputField()

    chunks = [
        _sc("[[ ## summary ## ]]\n\n"),
        _sc("This is a summary"),
        _sc(" of the response"),
        _sc("\n\n[[ ## details ## ]]\n\n"),
        _sc('{"message": "Detailed info"'),
        _sc(', "status": "complete"}'),
        _sc("\n\n[[ ## completed ## ]]"),
    ]

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="summary"),
            dspy.streaming.StreamListener(signature_field_name="details"),
        ],
    )

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Generate mixed response")
            summary_chunks = []
            details_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    if value.signature_field_name == "summary":
                        summary_chunks.append(value)
                    elif value.signature_field_name == "details":
                        details_chunks.append(value)

    assert len(summary_chunks) > 0
    assert len(details_chunks) > 0
    summary_content = "".join(chunk.chunk for chunk in summary_chunks)
    details_content = "".join(chunk.chunk for chunk in details_chunks)
    assert "summary" in summary_content
    assert "Detailed info" in details_content


@pytest.mark.anyio
async def test_json_adapter_simple_pydantic_streaming():
    """Test JSONAdapter streaming with a simple pydantic model."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    chunks = [
        _sc('{"'), _sc('response"'), _sc(":"),
        _sc('{"message"'), _sc(': "Hello'), _sc(' JSON!"'),
        _sc(', "status"'), _sc(': "ok"}'),
        _sc("}"),
    ]

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Say hello in JSON")
            chunks_out = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks_out.append(value)

    assert len(chunks_out) > 0
    assert chunks_out[0].signature_field_name == "response"
    full_content = "".join(chunk.chunk for chunk in chunks_out)
    assert "Hello JSON!" in full_content


@pytest.mark.anyio
async def test_json_adapter_bracket_balance_detection():
    """Test JSONAdapter correctly detects field completion using bracket balance."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: ComplexResponse = dspy.OutputField()

    chunks = [
        _sc('{"'), _sc('response": {'),
        _sc('"items": ["a"'), _sc(', "b"], '),
        _sc('"settings": {"key"'), _sc(': "value"}, '),
        _sc('"active": true}'),
        _sc("}"),
    ]

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Generate complex JSON")
            chunks_out = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks_out.append(value)

    assert len(chunks_out) > 0
    assert chunks_out[-1].is_last_chunk is True
    full_content = "".join(chunk.chunk for chunk in chunks_out)
    assert "items" in full_content
    assert "settings" in full_content


@pytest.mark.anyio
async def test_json_adapter_multiple_fields_detection():
    """Test JSONAdapter correctly detects when next field starts."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        first: SimpleResponse = dspy.OutputField()
        second: SimpleResponse = dspy.OutputField()

    chunks = [
        _sc('{"first": {'),
        _sc('"message": "first response"'),
        _sc(', "status": "ok"}'),
        _sc(', "second": {'),
        _sc('"message": "second response"'),
        _sc(', "status": "done"}}'),
    ]

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="first"),
            dspy.streaming.StreamListener(signature_field_name="second"),
        ],
    )

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Generate two responses")
            first_chunks = []
            second_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    if value.signature_field_name == "first":
                        first_chunks.append(value)
                    elif value.signature_field_name == "second":
                        second_chunks.append(value)

    assert len(first_chunks) > 0
    assert len(second_chunks) > 0
    first_content = "".join(chunk.chunk for chunk in first_chunks)
    second_content = "".join(chunk.chunk for chunk in second_chunks)
    assert "first response" in first_content
    assert "second response" in second_content


def test_stream_listener_could_form_end_identifier_chat_adapter():
    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    assert listener._could_form_end_identifier("some text [", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ #", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ##", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ## com", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ## completed", "ChatAdapter") is True
    assert listener._could_form_end_identifier("hello world", "ChatAdapter") is False
    assert listener._could_form_end_identifier("some text", "ChatAdapter") is False
    assert listener._could_form_end_identifier("answer: hello", "ChatAdapter") is False


def test_stream_listener_could_form_end_identifier_json_adapter():
    listener = dspy.streaming.StreamListener(signature_field_name="output")
    assert listener._could_form_end_identifier('some text "', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text ",', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text " ', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text "}', "JSONAdapter") is True
    assert listener._could_form_end_identifier("hello world", "JSONAdapter") is False
    assert listener._could_form_end_identifier("some text", "JSONAdapter") is False


def test_stream_listener_could_form_end_identifier_xml_adapter():
    listener = dspy.streaming.StreamListener(signature_field_name="result")
    assert listener._could_form_end_identifier("some text <", "XMLAdapter") is True
    assert listener._could_form_end_identifier("some text </", "XMLAdapter") is True
    assert listener._could_form_end_identifier("some text </result", "XMLAdapter") is True
    assert listener._could_form_end_identifier("hello world", "XMLAdapter") is False
    assert listener._could_form_end_identifier("some text", "XMLAdapter") is False


@pytest.mark.anyio
async def test_streaming_reasoning_model():
    """Test streaming behavior for reasoning-capable models using dspy.Reasoning."""

    class ReasoningSignature(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(ReasoningSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    chunks = [
        # Reasoning content comes through reasoning_content
        StreamChunk(reasoning_content="First, let's think about this problem step by step. "),
        StreamChunk(reasoning_content="We need to consider the context of a kitchen. "),
        StreamChunk(reasoning_content="The chicken likely wants to reach something on the other side."),
        # Regular answer content comes through content
        _sc("[[ ## answer ## ]]\n"),
        _sc("To"), _sc(" get"), _sc(" to"), _sc(" the"),
        _sc(" other"), _sc(" side"),
        _sc("!\n\n[[ ## completed ## ]]"),
    ]

    with mock.patch("dspy.clients._anthropic.astream_complete", side_effect=_mock_astream(chunks, model="claude-3-7-sonnet-20250219")):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="reasoning"),
                dspy.streaming.StreamListener(signature_field_name="answer"),
            ],
        )
        with dspy.context(
            lm=dspy.LM("anthropic/claude-3-7-sonnet-20250219", cache=False),
            adapter=dspy.ChatAdapter(native_response_types=[dspy.Reasoning]),
        ):
            output = program(question="Why did a chicken cross the kitchen?")
            reasoning_chunks = []
            answer_chunks = []
            final_prediction = None
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    if value.signature_field_name == "reasoning":
                        reasoning_chunks.append(value)
                    elif value.signature_field_name == "answer":
                        answer_chunks.append(value)
                elif isinstance(value, dspy.Prediction):
                    final_prediction = value

            # Verify reasoning chunks were streamed
            assert len(reasoning_chunks) == 3
            assert reasoning_chunks[0].chunk == "First, let's think about this problem step by step. "
            assert reasoning_chunks[1].chunk == "We need to consider the context of a kitchen. "
            assert reasoning_chunks[2].chunk == "The chicken likely wants to reach something on the other side."

            # Verify answer chunks were streamed
            assert len(answer_chunks) > 0
            assert answer_chunks[0].chunk == "To"
            full_answer = "".join([chunk.chunk for chunk in answer_chunks])
            assert full_answer == "To get to the other side!"

            # Verify final prediction has Reasoning object
            assert final_prediction is not None
            assert hasattr(final_prediction, "reasoning")
            assert isinstance(final_prediction.reasoning, dspy.Reasoning)
            expected_reasoning = (
                "First, let's think about this problem step by step. "
                "We need to consider the context of a kitchen. "
                "The chicken likely wants to reach something on the other side."
            )
            assert final_prediction.reasoning.content == expected_reasoning


@pytest.mark.anyio
async def test_stream_listener_empty_last_chunk_chat_adapter():
    predict = dspy.Predict("question->reasoning, answer")

    chunks = [
        _sc("[[ ## reasoning ## ]]\n"),
        _sc("Let's think about this problem step by step. "),
        _sc("We need to consider the context of a kitchen. "),
        _sc("The chicken likely wants to reach something on the other side. "),
        _sc("\n\n[[ ## answer ## ]]\n"),
        _sc("To get to the other side!"),
        _sc("\n\n[[ ## completed ## ]]"),
    ]

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        program = dspy.streamify(
            predict,
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="reasoning"),
                dspy.streaming.StreamListener(signature_field_name="answer"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Why did the chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

            reasoning_chunks = [c for c in all_chunks if c.signature_field_name == "reasoning"]
            answer_chunks = [c for c in all_chunks if c.signature_field_name == "answer"]

            assert answer_chunks[-1].is_last_chunk is True
            assert reasoning_chunks[-1].is_last_chunk is True


@pytest.mark.anyio
async def test_stream_listener_empty_last_chunk_json_adapter():
    predict = dspy.Predict("question->reasoning, answer")

    chunks = [
        _sc('{"reasoning": "'),
        _sc("Let's think about this problem step by step. "),
        _sc("We need to consider the context of a kitchen. "),
        _sc('The chicken likely wants to reach something on the other side. "'),
        _sc(',"answer": "'),
        _sc('To get to the other side!"'),
        _sc("\n}"),
    ]

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        program = dspy.streamify(
            predict,
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="reasoning"),
                dspy.streaming.StreamListener(signature_field_name="answer"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Why did the chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

            reasoning_chunks = [c for c in all_chunks if c.signature_field_name == "reasoning"]
            answer_chunks = [c for c in all_chunks if c.signature_field_name == "answer"]

            assert answer_chunks[-1].is_last_chunk is True
            assert reasoning_chunks[-1].is_last_chunk is True


@pytest.mark.anyio
async def test_streaming_reasoning_fallback():
    """Test fallback behavior for non-reasoning models using dspy.Reasoning."""

    class ReasoningSignature(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(ReasoningSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    chunks = [
        _sc("[[ ## reasoning ## ]]\n"),
        _sc("Let"), _sc("'s"), _sc(" think"), _sc(" step"), _sc(" by"),
        _sc(" step"), _sc(" about"), _sc(" this"), _sc(" question"), _sc("."),
        _sc("\n\n[[ ## answer ## ]]\n"),
        _sc("To"), _sc(" get"), _sc(" to"), _sc(" the"),
        _sc(" other"), _sc(" side"), _sc("!"),
        _sc("\n\n[[ ## completed ## ]]"),
    ]

    with mock.patch("dspy.clients._openai.astream_complete", side_effect=_mock_astream(chunks)):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="reasoning"),
                dspy.streaming.StreamListener(signature_field_name="answer"),
            ],
        )
        with dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False),
            adapter=dspy.ChatAdapter(),
        ):
            output = program(question="Why did a chicken cross the kitchen?")
            reasoning_chunks = []
            answer_chunks = []
            final_prediction = None
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    if value.signature_field_name == "reasoning":
                        reasoning_chunks.append(value)
                    elif value.signature_field_name == "answer":
                        answer_chunks.append(value)
                elif isinstance(value, dspy.Prediction):
                    final_prediction = value

            # Verify reasoning was streamed as regular text
            assert len(reasoning_chunks) > 0
            assert reasoning_chunks[0].chunk == "Let"
            assert reasoning_chunks[1].chunk == "'s"
            full_reasoning = "".join([chunk.chunk for chunk in reasoning_chunks])
            assert full_reasoning == "Let's think step by step about this question."

            # Verify answer chunks were streamed
            assert len(answer_chunks) > 0
            assert answer_chunks[0].chunk == "To"
            full_answer = "".join([chunk.chunk for chunk in answer_chunks])
            assert full_answer == "To get to the other side!"

            # Verify final prediction has Reasoning object created from string
            assert final_prediction is not None
            assert hasattr(final_prediction, "reasoning")
            assert isinstance(final_prediction.reasoning, dspy.Reasoning)
            assert final_prediction.reasoning.content == "Let's think step by step about this question."
            assert str(final_prediction.reasoning) == "Let's think step by step about this question."
