import json
from unittest import mock

import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import dspy


@pytest.mark.anyio
@pytest.mark.parametrize(
    "parts",
    [
        ['{"', "summary", '":', ' "', "Paris", " is", " big", '.",', ' "', "answer", '":', ' "', "Paris", '"', "}"],
        ['{"summary": "Paris is', ' big.", "answer": "Paris', '"}'],
        ['{"summary": "Paris is big.", "answer": "Paris"}'],
    ],
    ids=["token-sized", "multi-token", "single-chunk"],
)
async def test_json_stream_field_boundary(parts):
    await assert_streamed_fields(parts, {"summary": "Paris is big.", "answer": "Paris"})


@pytest.mark.anyio
@pytest.mark.parametrize("size", [1, 2, 7, 1000])
@pytest.mark.parametrize("summary", ["", "  Paris  ", 'quote: "answer", slash: \\, newline:\n, unicode: é 😀'])
async def test_json_stream_field_boundary_escaped_values(size, summary):
    expected = {"summary": summary, "answer": "Adjacent value"}
    response = json.dumps(expected)
    await assert_streamed_fields([response[i : i + size] for i in range(0, len(response), size)], expected)


async def assert_streamed_fields(parts, expected):
    async def stream(*args, **kwargs):
        for part in parts:
            yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=part))])

    program = dspy.streamify(
        dspy.Predict("question -> summary, answer"),
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name=field) for field in expected],
    )
    chunks = {field: [] for field in expected}
    final_chunks = dict.fromkeys(expected, 0)
    prediction = None
    with mock.patch("litellm.acompletion", side_effect=stream):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False), adapter=dspy.JSONAdapter()):
            async for value in program(question="q"):
                if isinstance(value, dspy.streaming.StreamResponse):
                    field = value.signature_field_name
                    chunks[field].append(value.chunk)
                    final_chunks[field] += value.is_last_chunk
                elif isinstance(value, dspy.Prediction):
                    prediction = value

    assert prediction is not None
    for field, expected_value in expected.items():
        assert getattr(prediction, field) == expected_value
    for field, expected_value in expected.items():
        # Every chunk must be a prefix extension of this field alone.
        for index in range(1, len(chunks[field]) + 1):
            assert expected_value.startswith("".join(chunks[field][:index]))
        assert "".join(chunks[field]) == expected_value
        assert final_chunks[field] == 1
