import os
import pytest

import dspy
from dspy.streaming import StatusMessage, StatusMessageProvider, streaming_response
from tests.test_utils.server import litellm_test_server


@pytest.mark.anyio
async def test_streamify_yields_expected_response_chunks(litellm_test_server):
    api_base, _ = litellm_test_server
    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):

        class TestSignature(dspy.Signature):
            input_text: str = dspy.InputField()
            output_text: str = dspy.OutputField()

        program = dspy.streamify(dspy.Predict(TestSignature))
        output_stream1 = program(input_text="Test")
        output_chunks1 = [chunk async for chunk in output_stream1]
        assert len(output_chunks1) > 1
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
            return f"Tool starting!"

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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found in environment variables")
@pytest.mark.anyio
async def test_stream_listener():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->judgement")

        def __call__(self, x: str, **kwargs):
            answer = self.predict1(question=x, **kwargs)
            judgement = self.predict2(question=x, answer=answer, **kwargs)
            return judgement

    # Turn off the cache to ensure the stream is produced.
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))
    my_program = MyProgram()
    program = dspy.streamify(
        my_program,
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer"),
            dspy.streaming.StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
    )
    output = program(x="why did a chicken cross the kitchen?")
    all_chunks = []
    async for value in output:
        if isinstance(value, dspy.streaming.StreamResponse):
            all_chunks.append(value)

    assert all_chunks[0].predict_name == "predict1"
    assert all_chunks[0].signature_field_name == "answer"

    assert all_chunks[-1].predict_name == "predict2"
    assert all_chunks[-1].signature_field_name == "judgement"


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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found in environment variables")
@pytest.mark.anyio
async def test_stream_listener_in_async_program():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->judgement")

        async def acall(self, x: str, **kwargs):
            answer = await self.predict1.acall(question=x, **kwargs)
            judgement = await self.predict2.acall(question=x, answer=answer, **kwargs)
            return judgement

    # Turn off the cache to ensure the stream is produced.
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))
    my_program = MyProgram()
    program = dspy.streamify(
        my_program,
        is_async_program=True,
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="answer"),
            dspy.streaming.StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
    )
    output = program(x="why did a chicken cross the kitchen?")
    all_chunks = []
    async for value in output:
        if isinstance(value, dspy.streaming.StreamResponse):
            all_chunks.append(value)

    assert all_chunks[0].predict_name == "predict1"
    assert all_chunks[0].signature_field_name == "answer"

    assert all_chunks[-1].predict_name == "predict2"
    assert all_chunks[-1].signature_field_name == "judgement"
