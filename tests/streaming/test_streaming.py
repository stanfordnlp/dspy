import asyncio
import time
from dataclasses import dataclass
from unittest import mock
from unittest.mock import AsyncMock

import anyio.from_thread
import pydantic
import pytest

import dspy
from dspy.adapters.types import Type
from dspy.experimental import Citations, Document
from dspy.streaming import StatusMessage, StatusMessageProvider, StreamResponse, streaming_response
from tests.test_utils.engines import patch_streaming

# Recorded OpenAI-shaped fixtures are plain wire dictionaries, not SDK models.
Delta = StreamingChoices = ModelResponseStream = dict


@pytest.mark.anyio
async def test_streamify_yields_expected_response_chunks(openai_test_server):
    api_base, _ = openai_test_server
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
async def test_streaming_response_yields_expected_response_chunks(openai_test_server):
    api_base, _ = openai_test_server
    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        cache=False,
    )
    # The server fixture speaks JSON. This encoding test must not depend on
    # replaying a response with another adapter after its chunks were emitted.
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):

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

    async def gpt_4o_mini_stream(*args, **kwargs):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]\n"))]
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="How "))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="are "))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="you "))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="doing?"))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    with patch_streaming(side_effect=gpt_4o_mini_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="What is the capital of France?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    assert "".join([chunk.chunk for chunk in all_chunks]) == "How are you doing?"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "parts, expected",
    [
        (["[[ ## answer ## ]]\n", "How ", "are ", "you ", "doing?\n\n", "[[ ## completed ## ]]"], "How are you doing?"),
        (["[[ ## answer ## ]]", "\nHow ", "are ", "you ", "doing?", "\n\n", "[[ ## completed ## ]]"], "How are you doing?"),
        (["[[ ## answer", " ## ]]", "\nHow ", "are ", "you ", "doing?", "\n\n[[ ## completed ## ]]"], "How are you doing?"),
        (["[[ ## answer ## ]]\n", "How\n", "are you\n", "doing?", "\n\n[[ ## completed ## ]]"], "How\nare you\ndoing?"),
        (["[[ ## answer ## ]]\n", "How ", "are ", "you ", "doing?\n\n"], "How are you doing?"),
    ],
)
async def test_streaming_strips_boundary_whitespace_regardless_of_chunking(parts, expected):
    program = dspy.streamify(
        dspy.Predict("question->answer"),
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    )

    async def stream(*args, **kwargs):
        for part in parts:
            yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=part))])

    with patch_streaming(side_effect=stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            all_chunks = []
            final = None
            async for value in program(question="What is the capital of France?"):
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)
                elif isinstance(value, dspy.Prediction):
                    final = value

    assert "".join([chunk.chunk for chunk in all_chunks]) == final.answer == expected


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


def test_apply_sync_streaming_propagates_generator_exceptions():
    class StreamFailureError(RuntimeError):
        pass

    async def failing_stream():
        yield "first"
        raise StreamFailureError("stream broke")

    sync_output = dspy.streaming.apply_sync_streaming(failing_stream())

    assert next(sync_output) == "first"
    with pytest.raises(StreamFailureError, match="stream broke"):
        next(sync_output)


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

    async def gpt_4o_mini_stream_1(*args, **kwargs):
        # Recorded streaming from openai/gpt-4o-mini
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[["))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="To"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" of"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" dinner"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" plate"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="!\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    async def gpt_4o_mini_stream_2():
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" judgement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="The"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" humorous"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" and"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" plays"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" on"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" classic"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" joke"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" format"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=".\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    stream_generators = [gpt_4o_mini_stream_1, gpt_4o_mini_stream_2]

    async def completion_side_effect(*args, **kwargs):
        return stream_generators.pop(0)()  # return new async generator instance

    with patch_streaming(side_effect=completion_side_effect):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
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

    async def gpt_4o_mini_stream_1(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='":'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"To'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" of"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" frying"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" pan"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='!"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])

    async def gpt_4o_mini_stream_2(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="jud"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="gement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='":'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"The'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" humorous"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" and"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" plays"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" on"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" very"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" funny"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" and"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" classic"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" joke"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" format"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='."'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])

    with patch_streaming(new_callable=AsyncMock, side_effect=[gpt_4o_mini_stream_1(), gpt_4o_mini_stream_2()]):
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

    async def gemini_stream_1(*args, **kwargs):
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content=" answer ## ]]"))])
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content="To get to the other side."))]
        )
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    async def gemini_stream_2(*args, **kwargs):
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content="[[ ## judgement ## ]]\n\n"))]
        )
        yield ModelResponseStream(
            model="gemini",
            choices=[
                StreamingChoices(
                    delta=Delta(
                        content=(
                            "The answer provides the standard punchline for this classic joke format, adapted to the "
                            "specific location mentioned in the question. It is the expected and appropriate response."
                        )
                    )
                )
            ],
        )
        yield ModelResponseStream(
            model="gemini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))],
        )
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="}\n"))])

    with patch_streaming(new_callable=AsyncMock, side_effect=[gemini_stream_1(), gemini_stream_2()]):
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
    """Test that streaming works correctly when LLM response omits a final completion marker.

    This test verifies that:
    1. All tokens are yielded including those in the buffer
    2. The last chunk is properly marked with is_last_chunk=True
    3. No tokens are lost when the completion marker is missing
    """

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def incomplete_stream(*args, **kwargs):
        """Stream that includes start marker but MISSING completion marker"""
        # Start marker
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])

        # Content tokens - more than 10 to ensure buffering happens
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="This"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" a"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" test"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" response"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" with"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" many"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" tokens"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ensure"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" buffering"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" works"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" correctly"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="."))])
        # NO COMPLETION MARKER

    with patch_streaming(side_effect=incomplete_stream):
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

    async def gemini_stream_1(*args, **kwargs):
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="{\n"))])
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content='  "answer": "To get to'))]
        )
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content=' the other side... of the cutting board!"'))]
        )
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="}\n"))])

    async def gemini_stream_2(*args, **kwargs):
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="{\n"))])
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content='  "judgement": "The'))]
        )
        yield ModelResponseStream(
            model="gemini",
            choices=[
                StreamingChoices(
                    delta=Delta(
                        content=' answer provides a humorous and relevant punchline to the classic joke setup."'
                    )
                )
            ],
        )
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="}\n"))])

    with patch_streaming(new_callable=AsyncMock, side_effect=[gemini_stream_1(), gemini_stream_2()]):
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

    with patch_streaming(new_callable=AsyncMock, side_effect=[dummy_tool]):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            timestamps = []
            async for value in output:
                if isinstance(value, dspy.streaming.StatusMessage):
                    timestamps.append(time.time())

    # timestamps[0]: tool start message
    # timestamps[1]: tool end message
    # There should be ~1 second delay between the tool start and end messages because we explicitly sleep for 1 second
    # in the tool.
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

    with patch_streaming(new_callable=AsyncMock, side_effect=[dummy_tool]):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            timestamps = []
            async for value in output:
                if isinstance(value, dspy.streaming.StatusMessage):
                    timestamps.append(time.time())

    # timestamps[0]: tool start message
    # timestamps[1]: tool end message
    # There should be ~1 second delay between the tool start and end messages because we explicitly sleep for 1 second
    # in the tool.
    assert timestamps[1] - timestamps[0] >= 1


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.XMLAdapter(use_json_adapter_fallback=False)])
async def test_stream_listener_allow_reuse(adapter):
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

    async def gpt_4o_mini_stream(*args, **kwargs):
        if isinstance(adapter, dspy.XMLAdapter):
            response = "```xml\n<answer>\nTo get to the other side!\n</answer>\n```"
            for start in range(0, len(response), 7):
                yield ModelResponseStream(
                    model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=response[start:start + 7]))]
                )
            return
        # Recorded streaming from openai/gpt-4o-mini
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[["))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="To"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="!\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    stream_generators = [gpt_4o_mini_stream, gpt_4o_mini_stream]

    async def completion_side_effect(*args, **kwargs):
        return stream_generators.pop(0)()  # return new async generator instance

    with patch_streaming(side_effect=completion_side_effect):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=adapter):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    concat_message = "".join([chunk.chunk for chunk in all_chunks])
    # The listener functions twice.
    assert concat_message == "To get to the other side!To get to the other side!"


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", [None, dspy.XMLAdapter()], ids=["default", "explicit-xml"])
async def test_stream_listener_returns_correct_chunk_xml_adapter(adapter):
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question,answer->judgement")

        def forward(self, question, **kwargs):
            answer = self.predict1(question=question, **kwargs).answer
            judgement = self.predict2(question=question, answer=answer, **kwargs)
            return judgement

    async def xml_stream_1(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="To"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="!"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="/answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])

    async def xml_stream_2(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="judgement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="The"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" humorous"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="."))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="/judgement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])

    stream_generators = [xml_stream_1, xml_stream_2]

    async def completion_side_effect(*args, **kwargs):
        return stream_generators.pop(0)()

    with patch_streaming(side_effect=completion_side_effect):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=adapter):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            final = None
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)
                elif isinstance(value, dspy.Prediction):
                    final = value

    assert final is not None and final.judgement == "The answer is humorous."
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
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
@pytest.mark.parametrize("terminal_content", [False, True])
@pytest.mark.parametrize(
    "response, expected",
    [
        (
            '<answer>\n  if a < b:\n    print("</answer>", "&amp;")  \n\n</answer>',
            '  if a < b:\n    print("</answer>", "&amp;")  \n',
        ),
        (
            "<answer>\r\n\r\n  </answer> is literal & text\r\n  end  \r\n</answer>\r\n",
            "\r\n  </answer> is literal & text\r\n  end  ",
        ),
        ("<answer>  compact < &amp; text  </answer>", "compact < &amp; text"),
        ("<answer>print('</answer>')</answer>", "print('</answer>')"),
        (
            "<answer>\nx = '''\n</answer>\nR&B\n'''\n</answer>",
            "x = '''\n</answer>\nR&B\n'''",
        ),
        ("<answer>\n\n</answer>", ""),
        ("<answer>\n```python\nSUBMIT(1353, 5)\n```</answer>", "```python\nSUBMIT(1353, 5)\n```"),
        ("<answer>\r\nR&B &amp; 雪</answer>\r\n", "R&B &amp; 雪"),
        (
            "<think>Ignore <answer>wrong</answer>.</think>\n<thinking>Check again.</thinking>\n"
            "<answer>\nKeep <think>literal</think> text</answer>",
            "Keep <think>literal</think> text",
        ),
        (
            "```xml\n<think>Ignore <answer>wrong</answer>.</think>\n<answer>\nright\n</answer>\n```",
            "right",
        ),
        (
            "<think>Ignore <answer>wrong</answer>.</think>\n```xml\n<answer>\nright\n</answer>\n```",
            "right",
        ),
        (
            "<think>outer</think>\n```xml\n<think><answer>wrong</answer></think>\n"
            "<answer>\nright\n</answer>\n```",
            "right",
        ),
        ("Here is the result:\n\n<answer>\nR&B &amp; 雪\n</answer>", "R&B &amp; 雪"),
        ("<answer>Paris</answer>\n\nLet me know if you need anything else.", "Paris"),
        ("<answer>  \r\n    Paris  \r\n </answer>  \r\nThanks!", "    Paris  "),
        ("<answer>\n\n</answer>\nThanks!", ""),
        (
            "<answer>\nx = '''\n</answer>\nR&B\n'''\n</answer>\nLet me know if you need anything else.",
            "x = '''\n</answer>\nR&B\n'''",
        ),
        (
            "<think>Check.</think>\n```xml\n<answer>\nParis\n</answer>\n```\nThanks!",
            "Paris",
        ),
        (
            '<answer>\n<html><style>.a { color: red; }</style>\n'
            '<body><div><div>R&B &amp; 雪</div></div><script>const x = {tag: "</answer>"};</script></body>\n'
            '</html></answer>',
            '<html><style>.a { color: red; }</style>\n'
            '<body><div><div>R&B &amp; 雪</div></div><script>const x = {tag: "</answer>"};</script></body>\n'
            '</html>',
        ),
    ],
)
async def test_xml_stream_preserves_literal_text(response, expected, chunk_size, terminal_content):
    program = dspy.streamify(
        dspy.Predict("question->answer"),
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer", allow_reuse=True)],
    )

    async def stream(*args, **kwargs):
        for start in range(0, len(response), chunk_size):
            yield ModelResponseStream(
                model="gpt-4o-mini",
                choices=[StreamingChoices(
                    delta=Delta(content=response[start : start + chunk_size]),
                    finish_reason="stop" if terminal_content and start + chunk_size >= len(response) else None,
                )],
            )

    with patch_streaming(side_effect=stream):
        with dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False),
            adapter=dspy.XMLAdapter(use_json_adapter_fallback=False),
        ):
            for _ in range(2):
                values = [value async for value in program(question="Copy the text verbatim.")]
                chunks = [value for value in values if isinstance(value, dspy.streaming.StreamResponse)]
                assert values[-1].answer == expected
                assert "".join(chunk.chunk for chunk in chunks) == expected
                assert sum(chunk.is_last_chunk for chunk in chunks) == 1
                if expected and chunk_size == 1:
                    assert len(chunks) > 2  # Emit text incrementally, not just once at the closing tag.


@pytest.mark.anyio
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
@pytest.mark.parametrize("finish_reason", [None, "stop"])
async def test_xml_stream_recovers_html_without_consuming_following_field(chunk_size, finish_reason):
    class Page(dspy.Signature):
        html: str = dspy.OutputField()
        count: int = dspy.OutputField()
        note: str = dspy.OutputField()

    html = "<!DOCTYPE html>\n<html><body>R&B &amp; 雪 <count/> <count>wrong</count></body>\n</html>"
    note = "A later field can also mention\n</html>"
    response = f"<html>\n{html}\n</html>\n<count>2</count>\n<note>\n{note}\n</note>\nHope this helps!"

    async def stream(*args, **kwargs):
        for start in range(0, len(response), chunk_size):
            yield ModelResponseStream(
                model="gpt-4o-mini",
                choices=[StreamingChoices(delta=Delta(content=response[start : start + chunk_size]))],
            )
        if finish_reason:
            yield ModelResponseStream(
                model="gpt-4o-mini",
                choices=[StreamingChoices(delta=Delta(), finish_reason=finish_reason)],
            )

    program = dspy.streamify(
        dspy.Predict(Page),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name=name) for name in ("html", "count", "note")
        ],
    )
    with patch_streaming(side_effect=stream):
        with dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False),
            adapter=dspy.XMLAdapter(use_json_adapter_fallback=False),
        ):
            values = [value async for value in program()]
    assert values[-1].html == html
    assert values[-1].count == 2
    assert values[-1].note == note
    for name, expected in [("html", html), ("count", "2"), ("note", note)]:
        chunks = [v for v in values if isinstance(v, dspy.streaming.StreamResponse) and v.signature_field_name == name]
        assert "".join(c.chunk for c in chunks) == expected
        assert sum(c.is_last_chunk for c in chunks) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
@pytest.mark.parametrize("variant", ["model", "union", "joined_close", "unknown_field", "literal_close"])
async def test_xml_stream_skips_nested_fields_with_top_level_names(chunk_size, variant):
    class Meta(pydantic.BaseModel):
        meta: str
        answer: str

    class Result(dspy.Signature):
        meta: Meta = dspy.OutputField()
        answer: str = dspy.OutputField()

    expected = {"meta": Meta(meta="inner", answer="wrong"), "answer": "right"}
    if variant == "literal_close":
        expected["meta"].meta = "inner\n</meta>\nmore"
    adapter = dspy.XMLAdapter(use_json_adapter_fallback=False)
    response = adapter.format_assistant_message_content(Result, expected)
    signature = Result
    if variant == "union":
        signature = Result.with_updated_fields("meta", type_=Meta | str)
    elif variant == "joined_close":
        response = response.replace("</meta>\n\n<answer>", "</meta><answer>")
    elif variant == "unknown_field":
        response = response.replace("inner\n</meta>", "inner\n</meta>\n<extra>ignored</extra>")

    async def stream(*args, **kwargs):
        for start in range(0, len(response), chunk_size):
            yield ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=response[start:start + chunk_size]))])

    program = dspy.streamify(dspy.Predict(signature), stream_listeners=[dspy.streaming.StreamListener("answer")])
    with patch_streaming(side_effect=stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=adapter):
            if variant == "unknown_field":
                with pytest.raises(dspy.utils.exceptions.AdapterParseError):
                    _ = [value async for value in program()]
                return
            values = [value async for value in program()]
    assert values[-1].meta == expected["meta"]
    assert values[-1].answer == "right"
    assert "".join(v.chunk for v in values if isinstance(v, StreamResponse)) == "right"


@pytest.mark.anyio
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("variant", ["valid", "missing_close", "extra_first"])
async def test_xml_stream_preserves_list_item_boundaries(chunk_size, compact, variant):
    class Item(pydantic.BaseModel):
        name: str
        score: float

    class Result(dspy.Signature):
        rows: list[Item] = dspy.OutputField()
        answer: str = dspy.OutputField()

    rows = [Item(name="a", score=1.1), Item(name="b", score=2.2)]
    adapter = dspy.XMLAdapter(use_json_adapter_fallback=False)
    response = adapter.format_assistant_message_content(Result, {"rows": rows, "answer": "right"})
    if variant == "missing_close":
        response = response.replace("</name>", "", 1)
    elif variant == "extra_first":
        response = response.replace("</name>", "</name>\n<extra>ignored</extra>", 1)
    if compact:
        response = response.replace("\n", "")
    response += "\nLet me know if you need anything else."

    async def stream(*args, **kwargs):
        for start in range(0, len(response), chunk_size):
            yield ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=response[start:start + chunk_size]))])

    program = dspy.streamify(dspy.Predict(Result), stream_listeners=[dspy.streaming.StreamListener("answer")])
    with patch_streaming(side_effect=stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=adapter):
            if variant != "valid":
                with pytest.raises(dspy.utils.exceptions.AdapterParseError):
                    _ = [value async for value in program()]
                return
            values = [value async for value in program()]
    assert values[-1].rows == rows
    assert values[-1].answer == "right"
    assert "".join(v.chunk for v in values if isinstance(v, StreamResponse)) == "right"


@pytest.mark.anyio
async def test_xml_stream_reuse_after_native_reasoning():
    class Twice(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.ChainOfThought("->answer")

        def forward(self):
            self.predict()
            return self.predict(config={"reasoning_effort": None})

    turns = iter([
        [Delta(reasoning_content="native thought"), Delta(content="<answer>one</answer>")],
        [Delta(content="<plan>fallback thought</plan><answer>two</answer>")],
    ])

    async def stream(*args, **kwargs):
        for delta in next(turns):
            yield ModelResponseStream(choices=[StreamingChoices(delta=delta)])

    program = dspy.streamify(Twice(), stream_listeners=[dspy.streaming.StreamListener("reasoning", allow_reuse=True)])
    with (
        patch_streaming(side_effect=stream),
        mock.patch.object(dspy.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=True),
        dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False),
            adapter=dspy.XMLAdapter(use_json_adapter_fallback=False),
        ),
    ):
        values = [value async for value in program()]
    assert values[-1].reasoning == "fallback thought"
    assert values[-1].answer == "two"
    assert "".join(v.chunk for v in values if isinstance(v, StreamResponse)) == "native thoughtfallback thought"


@pytest.mark.anyio
async def test_xml_stream_native_tools_ignore_incomplete_accompanying_thought():
    class Result(dspy.Signature):
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        calls: dspy.ToolCalls = dspy.OutputField()

    def search(query: str) -> str:
        return query

    call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query":"test"}'}}

    async def stream(*args, **kwargs):
        yield ModelResponseStream(choices=[StreamingChoices(delta=Delta(content="<think>Need to search"))])
        yield ModelResponseStream(choices=[StreamingChoices(delta=Delta(tool_calls=[call]), finish_reason="tool_calls")])

    program = dspy.streamify(dspy.Predict(Result), stream_listeners=[dspy.streaming.StreamListener("answer")])
    with (
        patch_streaming(side_effect=stream),
        mock.patch.object(dspy.LM, "supports_function_calling", new_callable=mock.PropertyMock, return_value=True),
        dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False),
            adapter=dspy.XMLAdapter(use_json_adapter_fallback=False, use_native_function_calling=True),
        ),
    ):
        values = [value async for value in program(tools=[dspy.Tool(search)])]
    assert not any(isinstance(value, StreamResponse) for value in values)
    assert values[-1].calls.tool_calls[0].name == "search"
    assert values[-1].calls.tool_calls[0].args == {"query": "test"}


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

            anyio.from_thread.run(send_to_stream)
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
            content = chunk.choices[0].delta.content
            return CustomType(message=content) if content is not None else None

        @classmethod
        def parse_lm_response(cls, response) -> "CustomType":
            return CustomType(message=response.text.split("\n\n")[0])

    class CustomSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: CustomType = dspy.OutputField()

    program = dspy.streamify(
        dspy.Predict(CustomSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer"),
        ],
    )

    async def stream(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="Hello"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="World"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    with patch_streaming(side_effect=stream):
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

    async def citation_stream(*args, **kwargs):
        # Stream chunks with citation data in provider_specific_fields
        # To verify the realistic scenario with more than 10 chunks in the stream, include more than 10 chunks before the citation.
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="A"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="c"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="c"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="o"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="r"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="d"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="i"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="n"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="g"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" to "))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="the references,"))])
        from dspy.lm15 import CitationDelta, StreamDeltaEvent

        yield StreamDeltaEvent(delta=CitationDelta(text="water boils at 100°C", title="Physics Facts", part_index=1))
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" water"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" boils"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" at"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" 100°C"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=".\n\n[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" citations ## ]]\n[]\n\n[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    # The provider's citation reaches the listener as a streamed part and the
    # final prediction as a CitationPart of the response.
    with patch_streaming(return_value=citation_stream()):
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
            assert final_prediction.citations[0].cited_text == "water boils at 100°C"
            assert final_prediction.citations[0].document_title == "Physics Facts"


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

    async def chat_stream(*args, **kwargs):
        # Simulate streaming of a pydantic model via ChatAdapter format
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" response"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message": "Hello'))]
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' world!"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status":'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' "success"}'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with patch_streaming(side_effect=chat_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Say hello")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)

    # Verify we got chunks for the pydantic field
    assert len(chunks) > 0
    assert chunks[0].signature_field_name == "response"

    # Combine all chunks to verify the content
    full_content = "".join(chunk.chunk for chunk in chunks)
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

    async def chat_stream(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" response"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="1"))]
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with patch_streaming(side_effect=chat_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Say hello")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)

    assert len(chunks) > 0
    assert chunks[0].signature_field_name == "response"

    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "1" in full_content


@pytest.mark.anyio
async def test_chat_adapter_nested_pydantic_streaming():
    """Test ChatAdapter streaming with nested pydantic model."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: NestedResponse = dspy.OutputField()

    async def nested_stream(*args, **kwargs):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## response ## ]]\n\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"title": "Test"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "content": {"key": "value"}'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "metadata": {"message": "nested"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "ok"}}'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with patch_streaming(side_effect=nested_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="Generate nested response")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)

    assert len(chunks) > 0
    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "nested" in full_content
    assert "Test" in full_content


@pytest.mark.anyio
async def test_chat_adapter_mixed_fields_streaming():
    """Test ChatAdapter streaming with both pydantic and string fields."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        summary: str = dspy.OutputField()
        details: SimpleResponse = dspy.OutputField()

    async def mixed_stream(*args, **kwargs):
        # First output field (summary - string)
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## summary ## ]]\n\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="This is a summary"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" of the response"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## details ## ]]\n\n"))]
        )
        # Second output field (details - pydantic)
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message": "Detailed info"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "complete"}'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="summary"),
            dspy.streaming.StreamListener(signature_field_name="details"),
        ],
    )

    with patch_streaming(side_effect=mixed_stream):
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

    # Verify both field types were streamed
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

    async def json_stream(*args, **kwargs):
        # Simulate JSON streaming with proper bracket balance tracking
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='response"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=":"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "Hello'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' JSON!"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "ok"}'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}"))]
        )  # Close main object

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with patch_streaming(side_effect=json_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Say hello in JSON")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)

    assert len(chunks) > 0
    assert chunks[0].signature_field_name == "response"

    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "Hello JSON!" in full_content


@pytest.mark.anyio
async def test_json_adapter_bracket_balance_detection():
    """Test JSONAdapter correctly detects field completion using bracket balance."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: ComplexResponse = dspy.OutputField()

    async def complex_json_stream(*args, **kwargs):
        # Test nested objects and arrays for bracket counting
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='response": {'))]
        )  # +1 bracket
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"items": ["a"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "b"], '))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"settings": {"key"'))]
        )  # +1 bracket
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "value"}, '))]
        )  # -1 bracket
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"active": true}'))]
        )  # -1 bracket (should end field)
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}"))]
        )  # Close main object

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with patch_streaming(side_effect=complex_json_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Generate complex JSON")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)

    assert len(chunks) > 0
    # Check that the last chunk is marked as the last
    assert chunks[-1].is_last_chunk is True

    full_content = "".join(chunk.chunk for chunk in chunks)

    assert "items" in full_content
    assert "settings" in full_content


@pytest.mark.anyio
async def test_json_adapter_multiple_fields_detection():
    """Test JSONAdapter correctly detects when next field starts."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        first: SimpleResponse = dspy.OutputField()
        second: SimpleResponse = dspy.OutputField()

    async def multi_field_stream(*args, **kwargs):
        # First field
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"first": {'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"message": "first response"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "ok"}'))]
        )
        # Second field starts
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "second": {'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"message": "second response"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "done"}}'))]
        )

    program = dspy.streamify(
        dspy.Predict(TestSignature),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="first"),
            dspy.streaming.StreamListener(signature_field_name="second"),
        ],
    )

    with patch_streaming(side_effect=multi_field_stream):
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

    # Verify both fields were detected and streamed
    assert len(first_chunks) > 0
    assert len(second_chunks) > 0

    first_content = "".join(chunk.chunk for chunk in first_chunks)
    second_content = "".join(chunk.chunk for chunk in second_chunks)

    assert "first response" in first_content
    assert "second response" in second_content


def test_stream_listener_could_form_end_identifier_chat_adapter():
    listener = dspy.streaming.StreamListener(signature_field_name="answer")

    # Should return True for partial bracket sequences
    assert listener._could_form_end_identifier("some text [", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ #", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ##", "ChatAdapter") is True

    # Should return True for partial field names after "[[ ##"
    assert listener._could_form_end_identifier("some text [[ ## com", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ## completed", "ChatAdapter") is True

    # Should return False for text that clearly cannot form the pattern
    assert listener._could_form_end_identifier("hello world", "ChatAdapter") is False
    assert listener._could_form_end_identifier("some text", "ChatAdapter") is False
    assert listener._could_form_end_identifier("answer: hello", "ChatAdapter") is False


def test_stream_listener_could_form_end_identifier_json_adapter():
    listener = dspy.streaming.StreamListener(signature_field_name="output")

    # Should return True for partial quote/brace sequences
    assert listener._could_form_end_identifier('some text "', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text ",', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text " ', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text "}', "JSONAdapter") is True

    # Should return False for text that cannot form the pattern
    assert listener._could_form_end_identifier("hello world", "JSONAdapter") is False
    assert listener._could_form_end_identifier("some text", "JSONAdapter") is False


def test_stream_listener_could_form_end_identifier_xml_adapter():
    listener = dspy.streaming.StreamListener(signature_field_name="result")

    # Should return True for partial closing tag
    assert listener._could_form_end_identifier("some text <", "XMLAdapter") is True
    assert listener._could_form_end_identifier("some text </", "XMLAdapter") is True
    assert listener._could_form_end_identifier("some text </result", "XMLAdapter") is True

    # Should return False for text that cannot form the pattern
    assert listener._could_form_end_identifier("hello world", "XMLAdapter") is False
    assert listener._could_form_end_identifier("some text", "XMLAdapter") is False


@pytest.mark.anyio
async def test_streaming_reasoning_model():
    """Test streaming behavior for reasoning-capable models using dspy.Reasoning.

    This test verifies that:
    1. Reasoning content is extracted from delta.reasoning_content in stream chunks
    2. Reasoning chunks are streamed independently from regular content
    3. The final prediction contains a Reasoning object with the full reasoning content
    """

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

    async def reasoning_stream(*args, **kwargs):
        """Simulate streaming from a reasoning model like Claude 3.7 Sonnet"""
        # Reasoning content comes through delta.reasoning_content
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[
                StreamingChoices(delta=Delta(reasoning_content="First, let's think about this problem step by step. "))
            ],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(reasoning_content="We need to consider the context of a kitchen. "))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[
                StreamingChoices(
                    delta=Delta(reasoning_content="The chicken likely wants to reach something on the other side.")
                )
            ],
        )
        # Regular answer content comes through delta.content
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]\n"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content="To"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" get"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" to"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" the"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" other"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" side"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content="!\n\n[[ ## completed ## ]]"))],
        )

    with patch_streaming(side_effect=reasoning_stream):
        with mock.patch.object(dspy.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=True):
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
    """Test that StreamListener emits an empty chunk marking field end.

    This test covers the scenario where:
    1. Tokens that cannot form the end identifier are immediately yielded
    2. The last chunk received contains only the marker for the next field (or completion marker)
    3. An empty chunk with is_last_chunk=True is emitted to properly mark field end
    """

    predict = dspy.Predict("question->reasoning, answer")

    async def mock_stream(*args, **kwargs):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## reasoning ## ]]\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="Let's think about this problem step by step. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="We need to consider the context of a kitchen. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[
                StreamingChoices(delta=Delta(content="The chicken likely wants to reach something on the other side. "))
            ],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## answer ## ]]\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="To get to the other side!"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))],
        )

    with patch_streaming(side_effect=mock_stream):
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

            # Find answer and judgement chunks
            reasoning_chunks = [c for c in all_chunks if c.signature_field_name == "reasoning"]
            answer_chunks = [c for c in all_chunks if c.signature_field_name == "answer"]

            # The last chunk should be marked as last chunk for both fields.
            assert answer_chunks[-1].is_last_chunk is True
            assert reasoning_chunks[-1].is_last_chunk is True


@pytest.mark.anyio
async def test_stream_listener_empty_last_chunk_json_adapter():
    predict = dspy.Predict("question->reasoning, answer")

    async def mock_stream(*args, **kwargs):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"reasoning": "'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="Let's think about this problem step by step. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="We need to consider the context of a kitchen. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[
                StreamingChoices(
                    delta=Delta(content='The chicken likely wants to reach something on the other side. "')
                )
            ],
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=',"answer": "'))])
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content='To get to the other side!"'))],
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n}"))])

    with patch_streaming(side_effect=mock_stream):
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

            # Find answer and judgement chunks
            reasoning_chunks = [c for c in all_chunks if c.signature_field_name == "reasoning"]
            answer_chunks = [c for c in all_chunks if c.signature_field_name == "answer"]

            # The last chunk should be marked as last chunk for both fields.
            assert answer_chunks[-1].is_last_chunk is True
            assert reasoning_chunks[-1].is_last_chunk is True


@pytest.mark.anyio
async def test_streaming_reasoning_fallback():
    """Test fallback behavior for non-reasoning models using dspy.Reasoning.

    This test verifies that:
    1. For non-reasoning models, reasoning is treated as a regular string field
    2. Reasoning content is streamed through regular adapter parsing (not reasoning_content)
    3. The Reasoning object is created from the parsed string content
    4. Streaming behavior is identical to regular string fields
    """

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

    async def non_reasoning_stream(*args, **kwargs):
        """Simulate streaming from a non-reasoning model like GPT-4o-mini.

        The reasoning field is formatted by the adapter as a regular field,
        and content comes through delta.content (not reasoning_content).
        """
        # Reasoning field marker (ChatAdapter format)
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="[[ ## reasoning ## ]]\n"))],
        )
        # Reasoning content as regular text
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="Let"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="'s"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" think"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" step"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" by"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" step"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" about"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" this"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" question"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="."))],
        )
        # Answer field marker
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## answer ## ]]\n"))],
        )
        # Answer content
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="To"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" get"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" to"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" the"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" other"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" side"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="!"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))],
        )

    with patch_streaming(side_effect=non_reasoning_stream):
        with mock.patch.object(dspy.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=False):
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
                # Verify Reasoning object is str-like
                assert str(final_prediction.reasoning) == "Let's think step by step about this question."


# ─── errors inside streamify are the program's own exception ─────────


class _FailingEngine:
    def complete(self, request):
        from dspy.lm15 import AuthError

        raise AuthError("bad key", provider="x")

    def stream(self, request):
        from dspy.lm15 import AuthError

        raise AuthError("bad key", provider="x")
        yield  # a generator that fails before its first event


def test_streamify_raises_the_public_error_itself():
    # The task group reported every failure as an exception group, so
    # `except dspy.LMError` missed it.
    lm = dspy.LM("custom/failing", engine=_FailingEngine(), cache=False, num_retries=0)
    stream = dspy.streamify(dspy.Predict("q -> a"), async_streaming=False)
    with dspy.context(lm=lm), pytest.raises(dspy.LMAuthError) as info:
        for _ in stream(q="x"):
            pass
    assert not hasattr(info.value, "exceptions")  # not a group


@pytest.mark.asyncio
async def test_async_streamify_raises_the_public_error_itself():
    lm = dspy.LM("custom/failing", engine=_FailingEngine(), cache=False, num_retries=0)
    stream = dspy.streamify(dspy.Predict("q -> a"), async_streaming=True)
    with dspy.context(lm=lm), pytest.raises(dspy.LMAuthError):
        async for _ in stream(q="x"):
            pass


@pytest.mark.anyio
@pytest.mark.parametrize("adapter_type", [dspy.ChatAdapter, dspy.JSONAdapter, dspy.XMLAdapter])
@pytest.mark.parametrize("mode", ["native", "unsupported", None, "none"])
async def test_default_cot_fallback_streams_reasoning(adapter_type, mode):
    texts = {
        dspy.ChatAdapter: "[[ ## plan ## ]]\nCount the signed items.\n\n[[ ## answer ## ]]\n17\n\n[[ ## completed ## ]]",
        dspy.JSONAdapter: '{"plan": "Count the signed items.", "answer": "17"}',
        dspy.XMLAdapter: "<plan>Count the signed items.</plan><answer>17</answer>",
    }
    if mode == "native":
        texts = {
            dspy.ChatAdapter: "[[ ## answer ## ]]\n17\n\n[[ ## completed ## ]]",
            dspy.JSONAdapter: '{"answer": "17"}',
            dspy.XMLAdapter: "<answer>17</answer>",
        }

    async def stream(*args, **kwargs):
        text = texts[adapter_type]
        for offset in range(0, len(text), 4):
            # Providers can return thinking even when disabled, including alongside content.
            yield ModelResponseStream(choices=[StreamingChoices(delta=Delta(
                content=text[offset:offset + 4], reasoning_content="Native thinking." if offset == 0 else None,
            ))])

    model = "openai/gpt-4o-mini" if mode == "unsupported" else "openai/gpt-5"
    lm = dspy.LM(model, cache=False, max_tokens=16000)
    lm.kwargs["reasoning_effort"] = "high"
    config = {"reasoning_effort": mode} if mode in (None, "none") else {}
    program = dspy.streamify(
        dspy.ChainOfThought("question -> answer"),
        stream_listeners=[dspy.streaming.StreamListener("reasoning")],
    )
    # Keep supported-but-disabled cases distinct from models without native reasoning.
    with (
        mock.patch.object(dspy.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=mode != "unsupported"),
        patch_streaming(side_effect=stream),
        dspy.context(lm=lm, adapter=adapter_type()),
    ):
        values = [value async for value in program(question="Count items", config=config)]
    chunks = [value for value in values if isinstance(value, StreamResponse)]
    assert chunks and all(c.signature_field_name == "reasoning" for c in chunks)
    expected = "Native thinking." if mode == "native" else "Count the signed items."
    streamed = "".join(c.chunk for c in chunks).strip()
    if adapter_type is dspy.JSONAdapter and mode != "native":
        # JSON listeners emit raw JSON fragments, not decoded strings.
        assert f'"{expected}"' in streamed
        assert "Native thinking." not in streamed
    else:
        assert streamed == expected
    assert values[-1].reasoning == expected
    assert values[-1].answer == "17"


@pytest.mark.anyio
async def test_xml_streaming_recognizes_a_renamed_reasoning_sibling():
    # Output order may differ from the prompt; `plan` is the adapter's wire name.
    text = "<answer>17</answer><plan>Count the signed items.</plan>"

    async def stream(*args, **kwargs):
        for offset in range(0, len(text), 4):
            yield ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=text[offset : offset + 4]))])

    program = dspy.streamify(
        dspy.ChainOfThought("question -> answer"),
        stream_listeners=[dspy.streaming.StreamListener(name) for name in ("answer", "reasoning")],
    )
    with (
        mock.patch.object(dspy.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=False),
        patch_streaming(side_effect=stream),
        dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()),
    ):
        values = [value async for value in program(question="Count items")]
    for name, expected in {"answer": "17", "reasoning": "Count the signed items."}.items():
        chunks = [
            value.chunk for value in values if isinstance(value, StreamResponse) and value.signature_field_name == name
        ]
        assert "".join(chunks) == expected
        assert getattr(values[-1], name) == expected


def test_streamify_keeps_a_group_of_several_failures():
    import builtins

    from dspy.streaming.streamify import _single_failure

    group_class = getattr(builtins, "BaseExceptionGroup", None)
    if group_class is None:  # Python 3.10: anyio's backport
        import exceptiongroup

        group_class = exceptiongroup.BaseExceptionGroup

    assert _single_failure(group_class("two", [ValueError("a"), ValueError("b")])) is None
    assert isinstance(_single_failure(group_class("outer", [group_class("inner", [KeyError("k")])])), KeyError)
    assert _single_failure(ValueError("plain")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streamify_mixes_nonchat_predictions_with_streamed_chat(asynchronous, monkeypatch):
    from dspy.clients.engines import AsyncLM15Engine, LM15Engine
    from dspy.lm15 import StreamDeltaEvent, StreamEndEvent, StreamStartEvent, TextDelta, Usage
    from tests.test_utils.engines import AsyncRecordingEngine, make_response

    calls = []
    text = "[[ ## city ## ]]\nParis\n\n[[ ## completed ## ]]"

    def complete(self, request):
        calls.append(request)
        return make_response(text, usage=Usage(input_tokens=9, output_tokens=4, total_tokens=13))

    async def acomplete(self, request):
        return complete(self, request)

    monkeypatch.setattr(LM15Engine, "complete", complete)
    monkeypatch.setattr(AsyncLM15Engine, "complete", acomplete)
    nonchat = dspy.LM("openai/fake", model_type="responses", cache=True, num_retries=0)

    class Chat:
        def complete(self, request):
            raise AssertionError("The chat prediction must still stream")

        def stream(self, request):
            yield StreamStartEvent(model=request.model)
            text = "[[ ## answer ## ]]\nEurope\n\n[[ ## completed ## ]]"
            for offset in range(0, len(text), 4):
                yield StreamDeltaEvent(TextDelta(text[offset:offset + 4]))
            yield StreamEndEvent(finish_reason="stop")

    engine = Chat()
    chat = dspy.LM("custom/chat", engine=engine, async_engine=AsyncRecordingEngine(engine), cache=True)

    class Program(dspy.Module):
        def __init__(self):
            self.helper = dspy.Predict("question -> city")
            self.helper.set_lm(nonchat)
            self.chat = dspy.Predict("city -> answer")
            self.chat.set_lm(chat)

        def forward(self, question):
            return self.chat(city=self.helper(question=question).city)

        async def aforward(self, question):
            city = (await self.helper.acall(question=question)).city
            return await self.chat.acall(city=city)

    # Only the selected predictor streams; its Responses-backed helper uses
    # complete(), regardless of the provider API's own streaming capability.
    program = dspy.streamify(Program(), is_async_program=asynchronous, stream_listeners=[
        dspy.streaming.StreamListener("answer"),
    ])
    with dspy.context(adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        for cached in (False, True):
            values = [value async for value in program(question="Capital of France?")]
            assert values[-1].answer == "Europe"
            chunks = [value for value in values if isinstance(value, StreamResponse)]
            assert all(chunk.signature_field_name == "answer" for chunk in chunks)
            assert "".join(chunk.chunk for chunk in chunks) == ("" if cached else "Europe")
    assert len(calls) == 1
    assert nonchat.history[0]["usage"]["total_tokens"] == 13
    assert nonchat.history[1]["usage"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("index_offset", [0, 7])
@pytest.mark.parametrize("direct", [False, True])
async def test_native_thinking_stream_preserves_text_and_replay(asynchronous, index_offset, direct):
    import json

    from dspy.lm15 import (
        AnthropicLM,
        ContinuationDelta,
        Message,
        Request,
        StreamDeltaEvent,
        StreamEndEvent,
        TextDelta,
        ThinkingDelta,
        ThinkingPart,
        Usage,
    )

    async def stream(*args, **kwargs):
        # Canonical events preserve distinct signed and redacted blocks,
        # even when the provider's part indices do not start at zero.
        for text in ("First ", "thought. "):
            yield StreamDeltaEvent(ThinkingDelta(text=text, part_index=index_offset))
        yield StreamDeltaEvent(ContinuationDelta(provider="anthropic", kind="thinking_signature",
            data={"signature": "sig-one"}, part_index=index_offset))
        yield StreamDeltaEvent(ThinkingDelta(text="", part_index=index_offset + 1))
        yield StreamDeltaEvent(ContinuationDelta(provider="anthropic", kind="redacted_thinking",
            data={"data": "opaque-redacted"}, part_index=index_offset + 1))
        yield StreamDeltaEvent(ThinkingDelta(text="Second thought.", part_index=index_offset + 2))
        yield StreamDeltaEvent(ContinuationDelta(provider="anthropic", kind="thinking_signature",
            data={"signature": "sig-two"}, part_index=index_offset + 2))
        texts = ("4", "2") if direct else ("[[ ## answer ## ]]\n", "42\n\n[[ ## completed ## ]]")
        for text in texts:
            yield StreamDeltaEvent(TextDelta(text=text, part_index=index_offset + 3))
        yield StreamEndEvent(finish_reason="stop", usage=Usage(input_tokens=9, output_tokens=4, total_tokens=13))

    lm = dspy.LM("anthropic/claude-sonnet-4-5", cache=True, num_retries=0)
    module = dspy.ChainOfThought("question -> answer")
    if direct:
        class Direct(dspy.Module):
            def forward(self, question):
                output = lm(question)[0]
                return dspy.Prediction(answer=output["text"], reasoning=output["reasoning_content"])

            async def aforward(self, question):
                output = (await lm.acall(question))[0]
                return dspy.Prediction(answer=output["text"], reasoning=output["reasoning_content"])

        module = Direct()
    program = dspy.streamify(module, is_async_program=asynchronous,
                            stream_listeners=[] if direct else [dspy.streaming.StreamListener("reasoning")])
    with patch_streaming(side_effect=stream), dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        for cached in (False, True):
            values = [value async for value in program(question="Think carefully")]
            assert values[-1].answer == "42"
            assert values[-1].reasoning == "First thought. Second thought."
            text = "".join(value.chunk for value in values if isinstance(value, StreamResponse))
            assert text == ("" if cached or direct else "First thought. Second thought.")
    response = lm.history[0]["response"]
    thoughts = response.message.parts_of(ThinkingPart)
    assert [part.text for part in thoughts] == ["First thought. ", "", "Second thought."]
    replay = Request(model="claude-sonnet-4-5", messages=(Message.user("Think"), response.message, Message.user("Continue")))
    body = AnthropicLM(api_key="test").build_request(replay, stream=False).body
    assert json.loads(body)["messages"][1]["content"][:3] == [
        {"type": "thinking", "thinking": "First thought. ", "signature": "sig-one"},
        {"type": "redacted_thinking", "data": "opaque-redacted"},
        {"type": "thinking", "thinking": "Second thought.", "signature": "sig-two"},
    ]
    assert lm.history[0]["usage"]["total_tokens"] == 13
    assert lm.history[1]["usage"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streamify_preserves_nonstreaming_responses_calls(asynchronous, monkeypatch):
    from dspy.clients.engines import AsyncLM15Engine, LM15Engine
    from dspy.lm15 import Usage
    from tests.test_utils.engines import make_response

    calls = []

    def complete(self, request):
        calls.append(request)
        return make_response("Paris", usage=Usage(input_tokens=9, output_tokens=4, total_tokens=13))

    async def acomplete(self, request):
        return complete(self, request)

    monkeypatch.setattr(LM15Engine, "complete", complete)
    monkeypatch.setattr(AsyncLM15Engine, "complete", acomplete)
    lm = dspy.LM("openai/fake", model_type="responses", cache=True, num_retries=0)

    class Program(dspy.Module):
        def forward(self, question):
            with dspy.context(send_stream=None):
                return dspy.Prediction(answer=lm(question)[0])

        async def aforward(self, question):
            with dspy.context(send_stream=None):
                return dspy.Prediction(answer=(await lm.acall(question))[0])

    program = dspy.streamify(Program(), is_async_program=asynchronous)
    for _ in range(2):
        values = [value async for value in program(question="Capital of France?")]
        assert len(values) == 1 and isinstance(values[0], dspy.Prediction)
        assert values[0].answer == {"text": "Paris"}
    assert len(calls) == 1
    assert lm.history[0]["usage"]["total_tokens"] == 13
    assert lm.history[1]["usage"] == {}
