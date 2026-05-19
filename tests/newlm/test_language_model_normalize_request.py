import pytest

import dspy


def make_lm():
    return dspy.BaseLM(model="test/model", cache=True, temperature=0.1)


def test_normalize_simple_text_positional_call():
    request = make_lm().normalize_request("hello")

    assert request == dspy.LMRequest(
        model="test/model",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(temperature=0.1, cache=dspy.lm.LMCacheConfig(enabled=True)),
    )


def test_normalize_single_message_positional_call():
    request = make_lm().normalize_request(dspy.User("hello"))

    assert request.messages == [dspy.User("hello")]


def test_normalize_prompt_keyword_call():
    request = make_lm().normalize_request(prompt="hello")

    assert request.messages == [dspy.User("hello")]


def test_normalize_text_with_generation_config():
    request = make_lm().normalize_request(
        "write one sentence about DSPy",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        stop=["END"],
        n=3,
        logprobs=True,
    )

    assert request.messages == [dspy.User("write one sentence about DSPy")]
    assert request.config.temperature == 0.7
    assert request.config.max_tokens == 100
    assert request.config.top_p == 0.9
    assert request.config.stop == ["END"]
    assert request.config.n == 3
    assert request.config.logprobs is True


def test_normalize_response_format_dict_and_pydantic_model():
    from pydantic import BaseModel
    class Person(BaseModel):
        name: str
        age: int

    dict_request = make_lm().normalize_request("Return JSON", response_format={"type": "json_object"})
    model_request = make_lm().normalize_request("Extract Alice, 32", response_format=Person)

    assert dict_request.config.response_format == {"type": "json_object"}
    assert model_request.config.response_format is Person


def test_normalize_reasoning_config_from_legacy_effort_kwarg():
    request = make_lm().normalize_request("think carefully", reasoning_effort="high")

    assert request.config.reasoning == dspy.lm.LMReasoningConfig(effort="high")


def test_normalize_reasoning_config_from_explicit_object():
    reasoning = dspy.lm.LMReasoningConfig(effort="high", summary="auto")

    request = make_lm().normalize_request("think carefully", reasoning=reasoning)

    assert request.config.reasoning == reasoning


def test_normalize_cache_config():
    request = make_lm().normalize_request("hello", cache=False, rollout_id=123, temperature=1.0)

    assert request.config.cache == dspy.lm.LMCacheConfig(enabled=False, rollout_id=123)
    assert request.config.temperature == 1.0


def test_normalize_cache_config_object_merges_rollout_id():
    request = make_lm().normalize_request("hello", cache=dspy.lm.LMCacheConfig(enabled=True), rollout_id=123)

    assert request.config.cache == dspy.lm.LMCacheConfig(enabled=True, rollout_id=123)


def test_normalize_provider_specific_kwargs_into_extensions():
    request = make_lm().normalize_request(
        "hello",
        service_tier="auto",
        extra_body={"provider_option": True},
        headers={"X-Trace-ID": "abc"},
    )

    assert request.config.extensions == {
        "service_tier": "auto",
        "extra_body": {"provider_option": True},
        "headers": {"X-Trace-ID": "abc"},
    }


def test_normalize_image_audio_file_and_reasoning_parts_in_one_user_message():
    request = make_lm().normalize_request(
        "summarize these inputs",
        dspy.Image("data:image/png;base64,image-bytes"),
        dspy.Audio(data="audio-bytes", audio_format="wav"),
        dspy.File(file_data="data:application/pdf;base64,file-bytes", filename="paper.pdf"),
        dspy.Reasoning("Prior reasoning supplied by the caller."),
    )

    assert request.messages == [
        dspy.User(
            "summarize these inputs",
            dspy.lm.LMImagePart(data="image-bytes", media_type="image/png"),
            dspy.lm.LMAudioPart(data="audio-bytes", media_type="audio/wav"),
            dspy.lm.LMFilePart(data="file-bytes", media_type="application/pdf", filename="paper.pdf"),
            dspy.lm.LMThinkingPart(text="Prior reasoning supplied by the caller."),
        )
    ]


def test_normalize_remote_image_url_part():
    request = make_lm().normalize_request("describe this", dspy.Image("https://example.com/dog.png"))

    assert request.messages == [
        dspy.User(
            "describe this",
            dspy.lm.LMImagePart(url="https://example.com/dog.png", media_type="image/png"),
        )
    ]


def test_normalize_openai_style_text_messages():
    request = make_lm().normalize_request(
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is DSPy?"},
        ],
        temperature=0.2,
    )

    assert request.messages == [
        dspy.System("You are concise."),
        dspy.User("What is DSPy?"),
    ]
    assert request.config.temperature == 0.2


def test_normalize_openai_style_multimodal_message_blocks():
    request = make_lm().normalize_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/dog.png"}},
                    {"type": "input_audio", "input_audio": {"data": "audio-bytes", "format": "wav"}},
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,file-bytes",
                            "filename": "paper.pdf",
                        },
                    },
                ],
            }
        ]
    )

    assert request.messages == [
        dspy.User(
            "describe this image",
            dspy.lm.LMImagePart(url="https://example.com/dog.png", media_type="image/png"),
            dspy.lm.LMAudioPart(data="audio-bytes", media_type="audio/wav"),
            dspy.lm.LMFilePart(data="file-bytes", media_type="application/pdf", filename="paper.pdf"),
        )
    ]


def test_normalize_message_constructors_for_multiturn():
    request = make_lm().normalize_request(
        dspy.System("You are concise."),
        dspy.Developer("Follow the house style."),
        dspy.User("What is DSPy?"),
        dspy.Assistant("DSPy is a framework for programming language model pipelines."),
        dspy.User("Say that in five words."),
    )

    assert request.messages == [
        dspy.System("You are concise."),
        dspy.Developer("Follow the house style."),
        dspy.User("What is DSPy?"),
        dspy.Assistant("DSPy is a framework for programming language model pipelines."),
        dspy.User("Say that in five words."),
    ]


def test_normalize_named_messages_list_is_canonical_conversation_form():
    request = make_lm().normalize_request(
        messages=[
            dspy.System("You are concise."),
            dspy.User("What is DSPy?"),
        ],
        temperature=0.2,
    )

    assert request.messages == [
        dspy.System("You are concise."),
        dspy.User("What is DSPy?"),
    ]
    assert request.config.temperature == 0.2


def test_normalize_single_inner_list_of_messages_is_one_conversation_not_batch():
    request = make_lm().normalize_request([
        dspy.System("You are concise."),
        dspy.User("What is DSPy?"),
    ])

    assert request.messages == [
        dspy.System("You are concise."),
        dspy.User("What is DSPy?"),
    ]


def test_normalize_bare_parts_are_one_implicit_user_message():
    request = make_lm().normalize_request(
        "Describe this image.",
        dspy.Image("https://example.com/dog.png"),
    )

    assert request.messages == [
        dspy.User(
            "Describe this image.",
            dspy.lm.LMImagePart(url="https://example.com/dog.png", media_type="image/png"),
        )
    ]


def test_normalize_rejects_ambiguous_positional_list_of_bare_parts():
    with pytest.raises(TypeError, match="Cannot convert"):
        make_lm().normalize_request(["hello", "world"])


def test_normalize_rejects_mixed_explicit_messages_and_bare_parts():
    with pytest.raises(TypeError, match="Cannot convert"):
        make_lm().normalize_request(
            dspy.System("You are concise."),
            dspy.User("Describe this image."),
            dspy.Image("https://example.com/dog.png"),
        )


def test_normalize_previous_lm_response_as_assistant_message():
    previous = dspy.LMResponse(
        model="test/model",
        outputs=[
            dspy.lm.LMOutput(
                parts=[
                    dspy.lm.LMThinkingPart(text="I should answer concisely."),
                    dspy.lm.LMTextPart(text="DSPy is a programming framework."),
                    dspy.lm.LMCitationPart(text="DSPy documentation", title="DSPy Docs", url="https://dspy.ai"),
                    dspy.lm.LMImagePart(data="image-bytes", media_type="image/png"),
                ]
            )
        ],
    )

    request = make_lm().normalize_request(
        dspy.User("What is DSPy?"),
        previous,
        dspy.User("Say that in five words."),
    )

    assert request.messages == [
        dspy.User("What is DSPy?"),
        dspy.Assistant(
            dspy.lm.LMThinkingPart(text="I should answer concisely."),
            "DSPy is a programming framework.",
            dspy.lm.LMCitationPart(text="DSPy documentation", title="DSPy Docs", url="https://dspy.ai"),
            dspy.lm.LMImagePart(data="image-bytes", media_type="image/png"),
        ),
        dspy.User("Say that in five words."),
    ]


def test_normalize_tool_call_and_tool_result_messages():
    request = make_lm().normalize_request(
        dspy.User("What is the weather in Paris?"),
        dspy.Assistant(dspy.LMToolCall(id="call_1", name="get_weather", args={"location": "Paris"})),
        dspy.ToolResult(
            call_id="call_1",
            name="get_weather",
            content=[
                "Here is the tool result: ",
                dspy.lm.LMTextPart(text='{"temperature": "22", "unit": "celsius"}'),
                dspy.Image("data:image/png;base64,weather-chart"),
            ],
        ),
        dspy.User("Summarize."),
    )

    assert request.messages == [
        dspy.User("What is the weather in Paris?"),
        dspy.Assistant(dspy.lm.LMToolCallPart(id="call_1", name="get_weather", args={"location": "Paris"})),
        dspy.ToolResult(
            dspy.lm.LMToolResultPart(
                call_id="call_1",
                name="get_weather",
                content=[
                    dspy.lm.LMTextPart(text="Here is the tool result: "),
                    dspy.lm.LMTextPart(text='{"temperature": "22", "unit": "celsius"}'),
                    dspy.lm.LMImagePart(data="weather-chart", media_type="image/png"),
                ],
            )
        ),
        dspy.User("Summarize."),
    ]


def test_normalize_bare_lm_parts_into_user_message():
    request = make_lm().normalize_request(
        dspy.lm.LMTextPart(text="describe this generated image"),
        dspy.lm.LMImagePart(data="generated-image", media_type="image/png"),
        dspy.lm.LMCitationPart(text="source text", title="Source", url="https://example.com/source"),
    )

    assert request.messages == [
        dspy.User(
            "describe this generated image",
            dspy.lm.LMImagePart(data="generated-image", media_type="image/png"),
            dspy.lm.LMCitationPart(text="source text", title="Source", url="https://example.com/source"),
        )
    ]


def test_normalize_positional_tool_sugar():
    def crop_image(x1: int, y1: int, x2: int, y2: int) -> str:
        """Crop the image to the given bounding box."""
        return "cropped-image-id"

    request = make_lm().normalize_request(
        "crop the dog",
        dspy.Image("data:image/png;base64,image-bytes"),
        dspy.Tool(crop_image),
    )

    assert request.messages == [
        dspy.User("crop the dog", dspy.lm.LMImagePart(data="image-bytes", media_type="image/png"))
    ]
    assert request.tools == [
        dspy.lm.LMToolSpec(
            name="crop_image",
            description="Crop the image to the given bounding box.",
            parameters={
                "type": "object",
                "properties": {
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                    "x2": {"type": "integer"},
                    "y2": {"type": "integer"},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        )
    ]


def test_normalize_explicit_tools_and_tool_choice():
    def search(query: str) -> str:
        """Search for a query."""
        return "result"

    request = make_lm().normalize_request(
        dspy.User("use search"),
        tools=[dspy.Tool(search)],
        tool_choice="required",
        parallel_tool_calls=False,
    )

    assert request.tools == [
        dspy.lm.LMToolSpec(
            name="search",
            description="Search for a query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    assert request.config.tool_choice == dspy.lm.LMToolChoice(mode="required", parallel=False)


def test_normalize_explicit_lm_request_and_kwarg_overrides():
    explicit = dspy.LMRequest(
        model="test/explicit",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(temperature=0.2, max_tokens=50),
    )

    request = make_lm().normalize_request(explicit, temperature=0.8)

    assert request.model == "test/explicit"
    assert request.messages == [dspy.User("hello")]
    assert request.config.temperature == 0.8
    assert request.config.max_tokens == 50


def test_explicit_lm_request_override_preserves_unspecified_config_fields():
    explicit = dspy.LMRequest(
        model="test/explicit",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(
            temperature=0.2,
            max_tokens=50,
            stop=["END"],
            extensions={"headers": {"X-Trace-ID": "abc"}},
            cache=dspy.lm.LMCacheConfig(enabled=True, rollout_id="old"),
            prompt_cache=dspy.lm.LMPromptCacheConfig(enabled=True, key="prefix-old"),
            tool_choice=dspy.lm.LMToolChoice(mode="auto", parallel=True),
            reasoning=dspy.lm.LMReasoningConfig(effort="low", max_tokens=100, summary="auto"),
        ),
    )

    request = make_lm().normalize_request(explicit, temperature=0.8)

    assert request.config.temperature == 0.8
    assert request.config.max_tokens == 50
    assert request.config.stop == ["END"]
    assert request.config.extensions == {"headers": {"X-Trace-ID": "abc"}}
    assert request.config.cache == dspy.lm.LMCacheConfig(enabled=True, rollout_id="old")
    assert request.config.prompt_cache == dspy.lm.LMPromptCacheConfig(enabled=True, key="prefix-old")
    assert request.config.tool_choice == dspy.lm.LMToolChoice(mode="auto", parallel=True)
    assert request.config.reasoning == dspy.lm.LMReasoningConfig(effort="low", max_tokens=100, summary="auto")


def test_explicit_lm_request_override_merges_extensions():
    explicit = dspy.LMRequest(
        model="test/explicit",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(extensions={"headers": {"X-Trace-ID": "abc"}}),
    )

    request = make_lm().normalize_request(explicit, service_tier="auto", extra_body={"provider_option": True})

    assert request.config.extensions == {
        "headers": {"X-Trace-ID": "abc"},
        "service_tier": "auto",
        "extra_body": {"provider_option": True},
    }


def test_explicit_lm_request_group_overrides_preserve_unspecified_subfields():
    explicit = dspy.LMRequest(
        model="test/explicit",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(
            cache=dspy.lm.LMCacheConfig(enabled=True, rollout_id="old"),
            prompt_cache=dspy.lm.LMPromptCacheConfig(enabled=True, key="prefix-old"),
            tool_choice=dspy.lm.LMToolChoice(mode="auto", parallel=True),
            reasoning=dspy.lm.LMReasoningConfig(effort="low", max_tokens=100, summary="auto"),
        ),
    )

    request = make_lm().normalize_request(
        explicit,
        rollout_id="new",
        prompt_cache_key="prefix-new",
        parallel_tool_calls=False,
        reasoning_effort="high",
    )

    assert request.config.cache == dspy.lm.LMCacheConfig(enabled=True, rollout_id="new")
    assert request.config.prompt_cache == dspy.lm.LMPromptCacheConfig(enabled=True, key="prefix-new")
    assert request.config.tool_choice == dspy.lm.LMToolChoice(mode="auto", parallel=False)
    assert request.config.reasoning == dspy.lm.LMReasoningConfig(effort="high", max_tokens=100, summary="auto")


def test_explicit_lm_request_override_can_clear_fields_with_none():
    explicit = dspy.LMRequest(
        model="test/explicit",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(temperature=0.2, stop=["END"], extensions={"headers": {"X": "1"}}),
    )

    request = make_lm().normalize_request(explicit, temperature=None, stop=None, extensions=None)

    assert request.config.temperature is None
    assert request.config.stop is None
    assert request.config.extensions == {}


def test_normalize_rejects_request_mixed_with_direct_inputs():
    explicit = dspy.LMRequest(model="test/explicit", messages=[dspy.User("hello")])

    with pytest.raises(ValueError, match="either an LMRequest or direct-call inputs"):
        make_lm().normalize_request(explicit, "extra text")


def test_normalize_rejects_messages_mixed_with_direct_inputs():
    with pytest.raises(ValueError, match="messages"):
        make_lm().normalize_request("hello", messages=[{"role": "user", "content": "hello"}])
