import os
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import dspy
from dspy.streaming import StatusMessage, StatusMessageProvider, streaming_response


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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found in environment variables")
@pytest.mark.anyio
async def test_stream_listener_chat_adapter():
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
async def test_stream_listener_json_adapter():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->judgement")

        def __call__(self, x: str, **kwargs):
            answer = self.predict1(question=x, **kwargs)
            judgement = self.predict2(question=x, answer=answer, **kwargs)
            return judgement

    # Turn off the cache to ensure the stream is produced.
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())
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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found in environment variables")
def test_sync_streaming():
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
        async_streaming=False,
    )
    output = program(x="why did a chicken cross the kitchen?")
    all_chunks = []
    for value in output:
        if isinstance(value, dspy.streaming.StreamResponse):
            all_chunks.append(value)

    assert all_chunks[0].predict_name == "predict1"
    assert all_chunks[0].signature_field_name == "answer"

    assert all_chunks[-1].predict_name == "predict2"
    assert all_chunks[-1].signature_field_name == "judgement"


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
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="!"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
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
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="."))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

    stream_generators = [gpt_4o_mini_stream_1, gpt_4o_mini_stream_2]

    async def completion_side_effect(*args, **kwargs):
        return stream_generators.pop(0)()  # return new async generator instance

    with mock.patch("litellm.acompletion", side_effect=completion_side_effect) as mock_completion:
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
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
        assert all_chunks[5].chunk == " side of the dinner plate!"

        # Start processing the second listened field.
        assert all_chunks[6].predict_name == "predict2"
        assert all_chunks[6].signature_field_name == "judgement"
        assert all_chunks[6].chunk == "The"
        assert all_chunks[7].chunk == " answer"
        assert all_chunks[8].chunk == " is"
        assert all_chunks[9].chunk == " humorous"
        assert all_chunks[10].chunk == " and"
        assert all_chunks[11].chunk == " plays"


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

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())

    async def gpt_4o_mini_stream_1(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='":'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="To"))])
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
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='":"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="The"))])
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
    ) as mock_completion:
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        output = program(question="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, dspy.streaming.StreamResponse):
                all_chunks.append(value)

        assert all_chunks[0].predict_name == "predict1"
        assert all_chunks[0].signature_field_name == "answer"

        assert all_chunks[0].chunk == "To"
        assert all_chunks[1].chunk == " get to the other side of the frying pan!"

        # Start processing the second listened field.
        assert all_chunks[2].predict_name == "predict2"
        assert all_chunks[2].signature_field_name == "judgement"
        assert all_chunks[2].chunk == "The"
        assert all_chunks[3].chunk == " answer"
        assert all_chunks[4].chunk == " is"
        assert all_chunks[5].chunk == " humorous"
        assert all_chunks[6].chunk == " and"
        assert all_chunks[7].chunk == " plays on the very funny and classic joke format."


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

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.ChatAdapter())

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
        output = program(question="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, dspy.streaming.StreamResponse):
                all_chunks.append(value)

        assert all_chunks[0].predict_name == "predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == "To get to the other side."

        assert all_chunks[1].predict_name == "predict2"
        assert all_chunks[1].signature_field_name == "judgement"
        assert all_chunks[1].chunk == (
            "The answer provides the standard punchline for this classic joke format, adapted to the specific location "
            "mentioned in the question. It is the expected and appropriate response."
        )


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

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())

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
        output = program(question="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, dspy.streaming.StreamResponse):
                all_chunks.append(value)

        assert all_chunks[0].predict_name == "predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == "To get to the other side... of the cutting board!"

        assert all_chunks[1].predict_name == "predict2"
        assert all_chunks[1].signature_field_name == "judgement"
        assert all_chunks[1].chunk == "The answer provides a humorous and relevant punchline to the classic joke setup."
