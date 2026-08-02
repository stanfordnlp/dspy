import asyncio
import time
from dataclasses import dataclass
from unittest import mock
from unittest.mock import AsyncMock

import anyio.from_thread
import pydantic
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import dspy
from dspy.adapters.types import Type
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.experimental import Citations, Document
from dspy.streaming import StatusMessage, StatusMessageProvider, StreamResponse, streaming_response
from dspy.utils.exceptions import AdapterParseError


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

    with mock.patch("litellm.acompletion", side_effect=gpt_4o_mini_stream):
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

    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):
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

    with mock.patch(
        "litellm.acompletion", new_callable=AsyncMock, side_effect=[gpt_4o_mini_stream_1(), gpt_4o_mini_stream_2()]
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

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[gemini_stream_1(), gemini_stream_2()]):
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

    with mock.patch("litellm.acompletion", side_effect=incomplete_stream):
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

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[gemini_stream_1(), gemini_stream_2()]):
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

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[dummy_tool]):
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

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[dummy_tool]):
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

    async def gpt_4o_mini_stream(*args, **kwargs):
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

    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    concat_message = "".join([chunk.chunk for chunk in all_chunks])
    # The listener functions twice.
    assert concat_message == "To get to the other side!To get to the other side!"


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        # The chunk that completes the opening tag also carries the nested value's inner closing
        # tag. Matching the closing tag directly would read that inner tag as the field's end.
        (["<answer><answer>inner</answer>", "</answer>", "<other>sib</other>"], "<answer>inner</answer>"),
        # The sibling's opening tag is split across chunks, and the field's own opening tag is
        # still part-way through arriving when the inner tags land.
        (["<answer>", "\n<answer>inner</answer>\n", "<ot", "her>sib</other>"], "<answer>inner"),
    ],
)
@pytest.mark.anyio
async def test_xml_adapter_nested_value_survives_awkward_chunk_boundaries(chunks, expected):
    """A nested value must still reach the consumer when tags straddle chunk boundaries.

    Both layouts used to emit nothing at all: the cache-hit shortcut saw a start identifier
    followed by some `</field>` and ended the stream, either on the value's inner closing tag or
    on an opening tag that was really part of the value.
    """

    class TwoFields(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        other: str = dspy.OutputField()

    async def xml_stream(*args, **kwargs):
        for token in chunks:
            yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=token))])

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(TwoFields),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            streamed = [v.chunk async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)]

    assert "".join(streamed).strip() == expected
    # And it still matches what parse returns for the same completion.
    assert dspy.XMLAdapter().parse(TwoFields, "".join(chunks))["answer"] == expected


@pytest.mark.anyio
async def test_xml_adapter_stream_surfaces_the_sibling_masked_ambiguity_from_parse():
    """A nested value truncated by a sibling inside its span must end in a loud error.

    The stream can only emit the lazy reading as tokens arrive -- a sent chunk cannot be
    recalled -- but the run must not conclude as if that truncated value were the answer:
    `parse` reports the ambiguity and the error reaches the stream consumer.
    """

    class TwoFields(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        other: str = dspy.OutputField()

    completion = "<answer><answer>nested</answer><other>sibling</other></answer>"

    async def xml_stream(*args, **kwargs):
        for i in range(0, len(completion), 3):
            yield ModelResponseStream(
                model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion[i : i + 3]))]
            )

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(TwoFields),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False),
            adapter=dspy.XMLAdapter(use_json_adapter_fallback=False),
        ):
            # The task group inside streamify re-raises as an exception group, so unwrap
            # rather than match the group type: the assertion is about the parse error.
            with pytest.raises(BaseException) as err:
                async for _ in program(question="?"):
                    pass

    # Unwrap by the group attribute rather than the type: `BaseExceptionGroup` is not a
    # builtin on Python 3.10, where anyio raises the `exceptiongroup` backport instead.
    def leaves(exc):
        subexceptions = getattr(exc, "exceptions", None)
        if subexceptions is None:
            return [exc]
        return [leaf for sub in subexceptions for leaf in leaves(sub)]

    assert any(isinstance(leaf, AdapterParseError) for leaf in leaves(err.value)), (
        f"expected an AdapterParseError, got {err.value!r}"
    )


@pytest.mark.anyio
async def test_xml_adapter_one_chunk_nested_value_lands_whole_in_the_prediction():
    """A nested value arriving in one chunk must land in the prediction untruncated.

    A completion that arrives whole in a single chunk takes the cache-hit path and emits no
    StreamResponse -- the pre-existing contract for every one-chunk response, plain or nested.
    What this pins is the value itself: the prediction must carry the full nested value, not
    the truncation the lazy scan used to produce.
    """

    class TwoFields(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        other: str = dspy.OutputField()

    completion = "<answer><answer>inner</answer></answer><other>sibling</other>"

    async def xml_stream(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion))])

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(TwoFields),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            streamed = []
            prediction = None
            async for value in program(question="?"):
                if isinstance(value, dspy.streaming.StreamResponse):
                    streamed.append(value.chunk)
                elif isinstance(value, dspy.Prediction):
                    prediction = value

    assert streamed == []
    assert prediction.answer == "<answer>inner</answer>"
    assert prediction.answer == dspy.XMLAdapter().parse(TwoFields, completion)["answer"]


@pytest.mark.anyio
async def test_xml_adapter_streams_nested_value_whole_and_matches_parse():
    """A nested same-named value must reach the consumer exactly as `parse` returns it.

    Which closing tag ends such a value is unknown until the balancing tag arrives, so the value
    is buffered and emitted once rather than token-by-token.
    """

    class CodeSignature(dspy.Signature):
        question: str = dspy.InputField()
        code: str = dspy.OutputField()

    completion = "<code>\n<code>x = 1</code>\n</code>"

    async def xml_stream(*args, **kwargs):
        for i in range(0, len(completion), 3):
            yield ModelResponseStream(
                model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion[i : i + 3]))]
            )

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(CodeSignature),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="code")],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            chunks = [v.chunk async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)]

    parsed = dspy.XMLAdapter().parse(CodeSignature, completion)["code"]
    assert parsed == "<code>x = 1</code>"
    assert "".join(chunks).strip() == parsed


@pytest.mark.anyio
async def test_xml_adapter_nested_stream_never_leaks_a_later_field():
    """Widening must stop at another output field rather than stream its value to the wrong listener."""

    class TwoFields(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        other: str = dspy.OutputField()

    # The `answer` span never balances, and `<other>` follows. Streaming `answer` must not emit it.
    completion = "<answer>\n<answer>foo\n</answer>\n<other>bar</other>"

    async def xml_stream(*args, **kwargs):
        for i in range(0, len(completion), 3):
            yield ModelResponseStream(
                model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion[i : i + 3]))]
            )

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(TwoFields),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            chunks = [v.chunk async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)]

    streamed = "".join(chunks).strip()
    assert "bar" not in streamed and "<other>" not in streamed
    assert streamed == dspy.XMLAdapter().parse(TwoFields, completion)["answer"]


@pytest.mark.anyio
async def test_xml_adapter_nested_stream_emits_at_the_boundary_not_at_finalize():
    """A buffered value must be released as soon as its boundary is settled, not once the stream ends.

    Here nothing balances the `answer` span, so what settles it is `<other>` opening -- which
    brings no closing tag. Waiting only on closing tags leaves the value stranded in the buffer
    until `finalize()`, arriving after the whole completion and never arriving at all when `parse`
    raises, since `finalize()` only runs once a `Prediction` is produced.
    """

    class TwoFields(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        other: str = dspy.OutputField()

    completion = "<answer>\n<answer>foo\n</answer>\n<other>bar baz qux quux</other>"

    async def xml_stream(*args, **kwargs):
        for i in range(0, len(completion), 3):
            yield ModelResponseStream(
                model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion[i : i + 3]))]
            )

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(TwoFields),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="other"),
            ],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            responses = [v async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)]

    order = [response.signature_field_name for response in responses]
    # `answer` lands before any of `other`'s tokens; a finalize-only emission would put it last.
    assert order.index("answer") < order.index("other")
    answer = "".join(r.chunk for r in responses if r.signature_field_name == "answer")
    assert answer == dspy.XMLAdapter().parse(TwoFields, completion)["answer"]


@pytest.mark.anyio
async def test_xml_adapter_nested_stream_decides_the_boundary_in_one_pass():
    """Settling the boundary must cost one pass over the value, however many children it holds.

    The decision used to start over at the top of the buffer every time a closing tag landed, so a
    nested value made of balanced children cost a pass per child; rebuilding the buffer to restart
    on then cost a copy of it per chunk, which is worse still, there being far more chunks than
    children. Both are quadratic, on the event loop, on exactly the payloads this path exists for.
    Counting passes bounds the first and counting the characters handed to the scan -- each one a
    character read and a character copied -- bounds the second, on any machine, where a wall-clock
    bound would only pin either on an unloaded one.
    """

    class CodeSignature(dspy.Signature):
        question: str = dspy.InputField()
        code: str = dspy.OutputField()

    decide = XMLAdapter._nested_decision_ready
    scan = XMLAdapter._nested_boundary_scan

    async def stream(children):
        completion = "<code>\n" + "".join(f"<code>{i}</code>" for i in range(children)) + "\n</code>"

        async def xml_stream(*args, **kwargs):
            for i in range(0, len(completion), 3):
                yield ModelResponseStream(
                    model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion[i : i + 3]))]
                )

        program = dspy.streamify(
            dspy.Predict(CodeSignature),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="code")],
        )
        with mock.patch("litellm.acompletion", side_effect=xml_stream):
            with mock.patch.object(XMLAdapter, "_nested_decision_ready", wraps=decide) as passes:
                with mock.patch.object(XMLAdapter, "_nested_boundary_scan", wraps=scan) as scans:
                    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
                        chunks = [
                            v.chunk async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)
                        ]
        return (
            passes.call_count,
            sum(len(call.args[1]) for call in scans.call_args_list) / len(completion),
            "".join(chunks).strip(),
            dspy.XMLAdapter().parse(CodeSignature, completion)["code"],
        )

    few_passes, few_reads, few_streamed, few_parsed = await stream(30)
    many_passes, many_reads, many_streamed, many_parsed = await stream(240)

    # Eight times the children, and the value still arrives whole -- so the counts below are not
    # those of a stream that gave up early.
    assert few_streamed == few_parsed
    assert many_streamed == many_parsed
    assert 1 <= few_passes == many_passes <= 2
    # Per character of the value, not in total: a bound that grows with the value is what a
    # quadratic scan would satisfy too. Restarting per chunk reads it once per chunk, so this ratio
    # would rise with the value rather than staying put.
    assert many_reads < 8 and many_reads < 2 * few_reads


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

    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):
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
async def test_xml_adapter_stream_decodes_escaped_value_and_matches_parse():
    """A compliant (escaped) wire must stream as the decoded value `parse` returns.

    The first `&lt;` arrives split across two chunks as `&l` | `t;`, so the decode has to hold the
    partial entity back rather than emit it raw or read it in halves.
    """

    class TwoFields(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        other: str = dspy.OutputField()

    chunks = ["<answer>", "\nif a &l", "t; b: print('&lt;/answer>')\n", "</answer>", "\n<other>ok</other>"]

    async def xml_stream(*args, **kwargs):
        for token in chunks:
            yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=token))])

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(TwoFields),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            streamed = [v.chunk async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)]

    value = "".join(streamed).strip()
    assert value == "if a < b: print('</answer>')"
    assert value == dspy.XMLAdapter().parse(TwoFields, "".join(chunks))["answer"]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5])
@pytest.mark.anyio
async def test_xml_adapter_stream_decode_is_chunking_invariant(chunk_size):
    """However the wire is chunked, the streamed value must equal what `parse` returns.

    The value below stacks entities back to back, including the escaped form of a literal `&lt;`,
    so every chunk size cuts some entity somewhere.
    """

    class TwoFields(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        other: str = dspy.OutputField()

    completion = "<answer>\na &amp;&amp; b &lt;&lt; c &amp;lt; d\n</answer>\n<other>ok</other>"

    async def xml_stream(*args, **kwargs):
        for i in range(0, len(completion), chunk_size):
            yield ModelResponseStream(
                model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion[i : i + chunk_size]))]
            )

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(TwoFields),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            streamed = [v.chunk async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)]

    assert "".join(streamed).strip() == "a && b << c &lt; d"
    assert "".join(streamed).strip() == dspy.XMLAdapter().parse(TwoFields, completion)["answer"]


@pytest.mark.anyio
async def test_xml_adapter_nested_stream_decodes_entities_like_parse():
    """The buffered nested-value emission must decode entities exactly as `parse` does."""

    class CodeSignature(dspy.Signature):
        question: str = dspy.InputField()
        code: str = dspy.OutputField()

    # A non-compliant wire: raw nested tags around an escaped `&`. The nested path buffers it and
    # must emit the decoded value, or the stream and the final Prediction would disagree.
    completion = "<code>\n<code>a &amp; b</code>\n</code>"

    async def xml_stream(*args, **kwargs):
        for i in range(0, len(completion), 3):
            yield ModelResponseStream(
                model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=completion[i : i + 3]))]
            )

    with mock.patch("litellm.acompletion", side_effect=xml_stream):
        program = dspy.streamify(
            dspy.Predict(CodeSignature),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="code")],
        )
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.XMLAdapter()):
            chunks = [v.chunk async for v in program(question="?") if isinstance(v, dspy.streaming.StreamResponse)]

    parsed = dspy.XMLAdapter().parse(CodeSignature, completion)["code"]
    assert parsed == "<code>a & b</code>"
    assert "".join(chunks).strip() == parsed


def test_xml_adapter_finalize_releases_a_held_entity_carry():
    """A partial entity held back mid-stream must come out, decoded, when the stream ends.

    The field-end queue is empty here -- the carry is all that is left -- so `finalize()` cannot
    rely on `flush()` alone; before the carry existed it returned None for an empty queue.
    """
    listener = dspy.streaming.StreamListener(signature_field_name="answer")

    def chunk(content):
        return ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=content))])

    with dspy.context(adapter=dspy.XMLAdapter()):
        emitted = []
        for content in ["<answer>", "x &am"]:
            response = listener.receive(chunk(content))
            if response is not None:
                emitted.append(response.chunk)
        final = listener.finalize()

    # Mid-stream the listener must not emit the half-entity `&am`...
    assert emitted == ["x "]
    # ...and at the end the tail has fully arrived: it was never an entity, so it comes out raw.
    assert final is not None and final.is_last_chunk
    assert final.chunk == "&am"


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

    async def stream(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="Hello"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="World"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    with mock.patch("litellm.acompletion", side_effect=stream):
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
        yield ModelResponseStream(
            model="claude",
            choices=[
                StreamingChoices(
                    delta=Delta(
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
                    )
                )
            ],
        )
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" water"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" boils"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" at"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" 100°C"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=".\n\n[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    # Mock the final response choice to include provider_specific_fields with citations
    with mock.patch("litellm.acompletion", return_value=citation_stream()):
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

    with mock.patch("litellm.acompletion", side_effect=chat_stream):
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
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="1"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=chat_stream):
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

    with mock.patch("litellm.acompletion", side_effect=nested_stream):
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

    with mock.patch("litellm.acompletion", side_effect=mixed_stream):
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

    with mock.patch("litellm.acompletion", side_effect=json_stream):
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

    with mock.patch("litellm.acompletion", side_effect=complex_json_stream):
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

    with mock.patch("litellm.acompletion", side_effect=multi_field_stream):
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

    with mock.patch("litellm.acompletion", side_effect=reasoning_stream):
        with mock.patch("litellm.supports_reasoning", return_value=True):
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

    with mock.patch("litellm.acompletion", side_effect=mock_stream):
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

    with mock.patch("litellm.acompletion", side_effect=mock_stream):
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

    with mock.patch("litellm.acompletion", side_effect=non_reasoning_stream):
        with mock.patch("litellm.supports_reasoning", return_value=False):
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
