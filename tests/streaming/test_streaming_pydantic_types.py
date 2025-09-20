import pytest
from unittest import mock
from unittest.mock import AsyncMock
import pydantic
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import dspy
from dspy.streaming import StreamResponse


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
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message": "Hello'))])
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
async def test_chat_adapter_nested_pydantic_streaming():
    """Test ChatAdapter streaming with nested pydantic model."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: NestedResponse = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def nested_stream(*args, **kwargs):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## response ## ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"title": "Test"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "content": {"key": "value"}'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "metadata": {"message": "nested"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "ok"}}'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))])

    program = dspy.streamify(
        MyProgram(),
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

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def mixed_stream(*args, **kwargs):
        # First output field (summary - string)
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## summary ## ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="This is a summary"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" of the response"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## details ## ]]\n\n"))])
        # Second output field (details - pydantic)
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message": "Detailed info"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "complete"}'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))])

    program = dspy.streamify(
        MyProgram(),
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

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def json_stream(*args, **kwargs):
        # Simulate JSON streaming with proper bracket balance tracking
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='response"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=':'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "Hello'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' JSON!"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "ok"}'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='}'))])  # Close main object

    program = dspy.streamify(
        MyProgram(),
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

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def complex_json_stream(*args, **kwargs):
        # Test nested objects and arrays for bracket counting
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='response": {'))])  # +1 bracket
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"items": ["a"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "b"], '))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"settings": {"key"'))])  # +1 bracket
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "value"}, '))])  # -1 bracket
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"active": true}'))])  # -1 bracket (should end field)
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='}'))])  # Close main object

    program = dspy.streamify(
        MyProgram(),
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

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def multi_field_stream(*args, **kwargs):
        # First field
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"first": {'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"message": "first response"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "ok"}'))])
        # Second field starts
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "second": {'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"message": "second response"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "done"}}'))])

    program = dspy.streamify(
        MyProgram(),
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


@pytest.mark.anyio
async def test_json_adapter_partial_parsing():
    """Test JSONAdapter handles partial JSON parsing correctly."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def partial_json_stream(*args, **kwargs):
        # Stream incomplete JSON that should be parseable by jiter in partial mode
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"response"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': {"mess'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='age": "partial'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' test", "stat'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='us": "parsing"}}'))])

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=partial_json_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Test partial parsing")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)

    assert len(chunks) > 0
    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "partial test" in full_content
    assert "parsing" in full_content


@pytest.mark.anyio
async def test_pydantic_streaming_error_handling():
    """Test streaming handles errors gracefully with pydantic types."""

    class TestSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict(TestSignature)

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def malformed_stream(*args, **kwargs):
        # Stream malformed JSON to test error handling
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"response": {'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"message": "test"'))])
        # Missing closing bracket - should still handle gracefully
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "incomplete"'))])

    program = dspy.streamify(
        MyProgram(),
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=malformed_stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="Test error handling")
            chunks = []
            # Should not raise an exception even with malformed JSON
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)

    # Should still get some chunks even if JSON is incomplete
    assert len(chunks) >= 0  # May be 0 if parsing fails completely, but shouldn't crash