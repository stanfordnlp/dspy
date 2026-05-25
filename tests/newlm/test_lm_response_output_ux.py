import dspy


def test_text_only_response_is_list_like_and_text_friendly():
    response = dspy.LMResponse.from_text(
        "Hello!",
        model="test/model",
        usage=dspy.LMUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        cost=0.00001,
    )

    assert response[0] == "Hello!"
    assert list(response) == ["Hello!"]
    assert response.output == response.outputs[0]
    assert response.parts == [dspy.LMTextPart(text="Hello!")]
    assert response.text == "Hello!"
    assert response.reasoning_content is None
    assert response.tool_calls == []
    assert response.citations == []
    assert response.images == []
    assert response.usage.total_tokens == 3
    assert response.cost == 0.00001
    assert response.cache_hit is False
    assert response.to_outputs() == ["Hello!"]


def test_text_plus_logprobs_keeps_output_level_metadata():
    logprobs = {"tokens": ["A"], "token_logprobs": [-0.1]}
    response = dspy.LMResponse(
        model="test/model",
        outputs=[dspy.LMOutput(parts=[dspy.LMTextPart(text="A")], logprobs=logprobs)],
    )

    assert response.text == "A"
    assert response.outputs[0].logprobs == logprobs
    assert response.to_outputs() == [{"text": "A", "logprobs": logprobs}]


def test_reasoning_text_and_tool_call_output_views():
    response = dspy.LMResponse(
        model="test/model",
        outputs=[
            dspy.LMOutput(
                parts=[
                    dspy.LMThinkingPart(text="I should call the weather tool."),
                    dspy.LMTextPart(text="I will check Paris now."),
                    dspy.LMToolCallPart(id="call_1", name="get_weather", args={"location": "Paris"}),
                ],
                finish_reason="tool_calls",
            )
        ],
    )

    assert response[0] == [
        dspy.Reasoning("I should call the weather tool."),
        "I will check Paris now.",
        dspy.ToolCall(id="call_1", name="get_weather", args={"location": "Paris"}),
    ]
    assert response.reasoning_content == "I should call the weather tool."
    assert response.text == "I will check Paris now."
    assert response.tool_calls == [
        dspy.LMToolCallPart(id="call_1", name="get_weather", args={"location": "Paris"})
    ]
    assert response.outputs[0].finish_reason == "tool_calls"
    assert response.to_outputs() == [
        {
            "text": "I will check Paris now.",
            "reasoning_content": "I should call the weather tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        }
    ]


def test_citations_are_projected_from_output_parts():
    citation = dspy.LMCitationPart(
        text="Water boils at 100°C.",
        title="Physics Handbook",
        url="https://example.com/physics",
    )
    response = dspy.LMResponse(
        model="test/model",
        outputs=[
            dspy.LMOutput(
                parts=[
                    dspy.LMTextPart(text="Water boils at 100°C."),
                    citation,
                ]
            )
        ],
    )

    assert response.text == "Water boils at 100°C."
    assert response.citations == [citation]
    assert response[0] == ["Water boils at 100°C.", citation]


def test_generated_images_audio_and_files_are_return_parts():
    image = dspy.LMImagePart(data="image-bytes", media_type="image/png")
    audio = dspy.LMAudioPart(data="audio-bytes", media_type="audio/wav")
    file = dspy.LMFilePart(data="file-bytes", media_type="application/pdf", filename="paper.pdf")
    response = dspy.LMResponse(
        model="test/model",
        outputs=[
            dspy.LMOutput(
                parts=[
                    dspy.LMTextPart(text="Here are the generated artifacts."),
                    image,
                    audio,
                    file,
                ]
            )
        ],
    )

    assert response.text == "Here are the generated artifacts."
    assert response.images == [image]
    assert response.audio == [audio]
    assert response.files == [file]
    assert response[0] == ["Here are the generated artifacts.", image, audio, file]


def test_multiple_outputs_keep_candidate_level_metadata_separate():
    response = dspy.LMResponse(
        model="test/model",
        outputs=[
            dspy.LMOutput(parts=[dspy.LMTextPart(text="first")], finish_reason="stop"),
            dspy.LMOutput(parts=[dspy.LMTextPart(text="second")], finish_reason="length", truncated=True),
            dspy.LMOutput(
                parts=[dspy.LMToolCallPart(id="call_1", name="search", args={"query": "DSPy"})],
                finish_reason="tool_calls",
            ),
        ],
    )

    assert list(response) == [
        "first",
        "second",
        [dspy.ToolCall(id="call_1", name="search", args={"query": "DSPy"})],
    ]
    assert response.outputs[0].finish_reason == "stop"
    assert response.outputs[1].truncated is True
    assert response.outputs[2].tool_calls == [
        dspy.LMToolCallPart(id="call_1", name="search", args={"query": "DSPy"})
    ]
    assert response.to_outputs() == [
        "first",
        "second",
        {
            "text": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "DSPy"}'},
                }
            ],
        },
    ]


def test_output_convenience_properties_are_candidate_local():
    output = dspy.LMOutput(
        parts=[
            dspy.LMThinkingPart(text="Think 1. "),
            dspy.LMThinkingPart(text="Think 2."),
            dspy.LMTextPart(text="Answer "),
            dspy.LMTextPart(text="text."),
            dspy.LMToolCallPart(id="call_1", name="lookup", args={"id": 1}),
            dspy.LMCitationPart(text="quoted", title="Source"),
            dspy.LMImagePart(data="image", media_type="image/png"),
        ]
    )

    assert output.reasoning_content == "Think 1. Think 2."
    assert output.text == "Answer text."
    assert output.tool_calls == [dspy.LMToolCallPart(id="call_1", name="lookup", args={"id": 1})]
    assert output.citations == [dspy.LMCitationPart(text="quoted", title="Source")]
    assert output.images == [dspy.LMImagePart(data="image", media_type="image/png")]


def test_usage_as_dict_omits_none_values_and_cache_hit_has_no_usage_or_cost():
    response = dspy.LMResponse.from_text(
        "cached",
        model="test/model",
        usage=None,
        cost=None,
        cache_hit=True,
    )

    assert response.cache_hit is True
    assert response.usage is None
    assert response.cost is None
    assert response.usage_as_dict() == {}


def test_streaming_events_assemble_into_lm_output_and_lm_response():
    builder = dspy.LMOutputBuilder()

    builder.apply(dspy.LMStreamStartEvent(model="test/model"))
    builder.apply(dspy.LMStreamDeltaEvent(output_index=0, part_index=0, delta=dspy.LMThinkingDelta(text="Think. ")))
    builder.apply(dspy.LMStreamDeltaEvent(output_index=0, part_index=1, delta=dspy.LMTextDelta(text="Hello")))
    builder.apply(dspy.LMStreamDeltaEvent(output_index=0, part_index=1, delta=dspy.LMTextDelta(text="!")))
    builder.apply(
        dspy.LMStreamDeltaEvent(
            output_index=0,
            part_index=2,
            delta=dspy.LMToolCallDelta(id="call_1", name="search", args_delta='{"query": "DSPy"}'),
        )
    )
    builder.apply(dspy.LMStreamOutputEndEvent(output_index=0, finish_reason="tool_calls"))
    final = builder.apply(
        dspy.LMStreamEndEvent(
            usage=dspy.LMUsage(input_tokens=5, output_tokens=7, total_tokens=12),
            cost=0.0002,
        )
    )

    assert final == dspy.LMResponse(
        model="test/model",
        outputs=[
            dspy.LMOutput(
                parts=[
                    dspy.LMThinkingPart(text="Think. "),
                    dspy.LMTextPart(text="Hello!"),
                    dspy.LMToolCallPart(
                        id="call_1",
                        name="search",
                        args={"query": "DSPy"},
                        provider_data={"args_buffer": '{"query": "DSPy"}'},
                    ),
                ],
                finish_reason="tool_calls",
            )
        ],
        usage=dspy.LMUsage(input_tokens=5, output_tokens=7, total_tokens=12),
        cost=0.0002,
    )
