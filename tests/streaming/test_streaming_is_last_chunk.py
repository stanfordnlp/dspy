"""Comprehensive tests for is_last_chunk behavior across all adapters.

These tests ensure that:
1. Every stream listener always yields at least one chunk with is_last_chunk=True
2. is_last_chunk=False for all non-final chunks
3. is_last_chunk=True only appears on the final chunk for each listener
4. This behavior is consistent across ChatAdapter, JSONAdapter, and XMLAdapter
5. This behavior works with both complete and incomplete completion markers
"""

from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litellm import ModelResponseStream
from litellm.types.utils import Delta, StreamingChoices

import dspy


@pytest.mark.anyio
async def test_is_last_chunk_always_present_chat_adapter():
    """Test that ChatAdapter always yields a final chunk with is_last_chunk=True."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def stream_with_completion_marker(*args, **kwargs):
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="Hello"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content=" world"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## completed ## ]]"))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[stream_with_completion_marker()]):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Verify we have at least one chunk
    assert len(all_chunks) > 0

    # Verify exactly one chunk has is_last_chunk=True and it's the last one
    last_chunk_count = sum(1 for chunk in all_chunks if chunk.is_last_chunk)
    assert last_chunk_count == 1, f"Expected exactly 1 chunk with is_last_chunk=True, got {last_chunk_count}"
    assert all_chunks[-1].is_last_chunk is True, "Last chunk must have is_last_chunk=True"

    # Verify all non-last chunks have is_last_chunk=False
    for chunk in all_chunks[:-1]:
        assert chunk.is_last_chunk is False, f"Non-final chunk should have is_last_chunk=False: {chunk}"


@pytest.mark.anyio
async def test_is_last_chunk_always_present_json_adapter():
    """Test that JSONAdapter always yields a final chunk with is_last_chunk=True."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def json_stream(*args, **kwargs):
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content='{"answer": "'))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="Hello"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content=" world"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content='"}'))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[json_stream()]):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.JSONAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Verify we have at least one chunk
    assert len(all_chunks) > 0

    # Verify exactly one chunk has is_last_chunk=True and it's the last one
    last_chunk_count = sum(1 for chunk in all_chunks if chunk.is_last_chunk)
    assert last_chunk_count == 1, f"Expected exactly 1 chunk with is_last_chunk=True, got {last_chunk_count}"
    assert all_chunks[-1].is_last_chunk is True, "Last chunk must have is_last_chunk=True"

    # Verify all non-last chunks have is_last_chunk=False
    for chunk in all_chunks[:-1]:
        assert chunk.is_last_chunk is False, f"Non-final chunk should have is_last_chunk=False: {chunk}"


@pytest.mark.anyio
async def test_is_last_chunk_always_present_xml_adapter():
    """Test that XMLAdapter always yields a final chunk with is_last_chunk=True."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def xml_stream(*args, **kwargs):
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="<answer>"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="Hello"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content=" world"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="</answer>"))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[xml_stream()]):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.XMLAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Verify we have at least one chunk
    assert len(all_chunks) > 0

    # Verify exactly one chunk has is_last_chunk=True and it's the last one
    last_chunk_count = sum(1 for chunk in all_chunks if chunk.is_last_chunk)
    assert last_chunk_count == 1, f"Expected exactly 1 chunk with is_last_chunk=True, got {last_chunk_count}"
    assert all_chunks[-1].is_last_chunk is True, "Last chunk must have is_last_chunk=True"

    # Verify all non-last chunks have is_last_chunk=False
    for chunk in all_chunks[:-1]:
        assert chunk.is_last_chunk is False, f"Non-final chunk should have is_last_chunk=False: {chunk}"


@pytest.mark.anyio
async def test_is_last_chunk_multiple_listeners_all_get_final_chunk():
    """Test that with multiple listeners, each gets its own final chunk with is_last_chunk=True."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question,answer->judgement")

        def forward(self, question, **kwargs):
            answer = self.predict1(question=question, **kwargs).answer
            judgement = self.predict2(question=question, answer=answer, **kwargs)
            return judgement

    async def stream_1(*args, **kwargs):
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="Answer"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## completed ## ]]"))])

    async def stream_2(*args, **kwargs):
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## judgement ## ]]"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="Judgement"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## completed ## ]]"))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[stream_1(), stream_2()]):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer"),
                dspy.streaming.StreamListener(signature_field_name="judgement"),
            ],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Separate chunks by listener
    answer_chunks = [c for c in all_chunks if c.signature_field_name == "answer"]
    judgement_chunks = [c for c in all_chunks if c.signature_field_name == "judgement"]

    # Each listener should have exactly one chunk with is_last_chunk=True
    assert len(answer_chunks) > 0, "Should have answer chunks"
    assert len(judgement_chunks) > 0, "Should have judgement chunks"

    answer_last_count = sum(1 for c in answer_chunks if c.is_last_chunk)
    judgement_last_count = sum(1 for c in judgement_chunks if c.is_last_chunk)

    assert answer_last_count == 1, (
        f"Answer listener should have exactly 1 chunk with is_last_chunk=True, got {answer_last_count}"
    )
    assert judgement_last_count == 1, (
        f"Judgement listener should have exactly 1 chunk with is_last_chunk=True, got {judgement_last_count}"
    )

    # The final chunk for each listener should be the last one
    assert answer_chunks[-1].is_last_chunk is True, "Last answer chunk must have is_last_chunk=True"
    assert judgement_chunks[-1].is_last_chunk is True, "Last judgement chunk must have is_last_chunk=True"


@pytest.mark.anyio
async def test_is_last_chunk_with_few_tokens():
    """Test that even responses with few tokens have is_last_chunk=True."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def few_tokens_stream(*args, **kwargs):
        # Just a few tokens
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="OK"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## completed ## ]]"))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[few_tokens_stream()]):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Should have at least one chunk with is_last_chunk=True
    assert len(all_chunks) > 0, "Should have at least one chunk"
    assert any(c.is_last_chunk for c in all_chunks), "At least one chunk should have is_last_chunk=True"
    assert all_chunks[-1].is_last_chunk is True, "Last chunk must have is_last_chunk=True"


@pytest.mark.anyio
async def test_is_last_chunk_order_invariant():
    """Test that is_last_chunk appears exactly once per listener regardless of chunk ordering."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            return self.predict(question=question, **kwargs)

    async def varied_chunk_sizes_stream(*args, **kwargs):
        # Mix of different sized chunks
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="A"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content=""))])  # Empty chunk
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="BC"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="DEF"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content=""))])  # Another empty
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## completed ## ]]"))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[varied_chunk_sizes_stream()]):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Count chunks with is_last_chunk=True
    last_chunk_count = sum(1 for chunk in all_chunks if chunk.is_last_chunk)
    assert last_chunk_count == 1, f"Should have exactly 1 chunk with is_last_chunk=True, got {last_chunk_count}"
    assert all_chunks[-1].is_last_chunk is True, "Last chunk must have is_last_chunk=True"


@pytest.mark.anyio
async def test_is_last_chunk_with_allow_reuse():
    """Test that is_last_chunk works correctly when allow_reuse=True and listener is used multiple times."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            # Call predict twice to test reusability
            self.predict(question=question, **kwargs)
            return self.predict(question=question, **kwargs)

    async def stream_call(*args, **kwargs):
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="Response"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## completed ## ]]"))])

    stream_generators = [stream_call, stream_call]

    async def completion_side_effect(*args, **kwargs):
        return stream_generators.pop(0)()

    # Test with allow_reuse=True
    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer", allow_reuse=True),
            ],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Should have chunks from both predict calls
    assert len(all_chunks) > 0, "Should have chunks from reused listener"

    # Count how many times is_last_chunk=True appears
    last_chunk_count = sum(1 for chunk in all_chunks if chunk.is_last_chunk)

    # With allow_reuse=True, the listener should be used for BOTH predict calls
    # So we should see TWO final chunks (one for each predict call)
    assert last_chunk_count == 2, (
        f"With allow_reuse=True, should have 2 final chunks (one per call), got {last_chunk_count}"
    )

    # Verify the listener captured both responses
    concat_message = "".join([chunk.chunk for chunk in all_chunks if chunk.chunk])
    assert "Response" in concat_message, "Should have captured response content"


@pytest.mark.anyio
async def test_is_last_chunk_without_allow_reuse():
    """Test that is_last_chunk works correctly when allow_reuse=False (default) and listener stops after first use."""

    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question->answer")

        def forward(self, question, **kwargs):
            # Call predict twice - but with allow_reuse=False, only first should be captured
            self.predict(question=question, **kwargs)
            return self.predict(question=question, **kwargs)

    async def stream_call(*args, **kwargs):
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="Response"))])
        yield ModelResponseStream(model="gpt", choices=[StreamingChoices(delta=Delta(content="[[ ## completed ## ]]"))])

    stream_generators = [stream_call, stream_call]

    async def completion_side_effect(*args, **kwargs):
        return stream_generators.pop(0)()

    # Test with allow_reuse=False (default)
    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):
        program = dspy.streamify(
            MyProgram(),
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="answer", allow_reuse=False),
            ],
        )
        with dspy.context(lm=dspy.LM("gpt", cache=False), adapter=dspy.ChatAdapter()):
            output = program(question="test")
            all_chunks = []
            async for value in output:
                if isinstance(value, dspy.streaming.StreamResponse):
                    all_chunks.append(value)

    # Should have chunks from only the first predict call
    assert len(all_chunks) > 0, "Should have chunks from first call"

    # Count how many times is_last_chunk=True appears
    last_chunk_count = sum(1 for chunk in all_chunks if chunk.is_last_chunk)

    # With allow_reuse=False (default), the listener should stop after first use
    # So we should see only ONE final chunk
    assert last_chunk_count == 1, f"With allow_reuse=False, should have 1 final chunk, got {last_chunk_count}"
    assert all_chunks[-1].is_last_chunk is True, "Last chunk must have is_last_chunk=True"
