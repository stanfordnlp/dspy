import dspy


class EchoLM(dspy.BaseLM):
    """Tiny normalized LM used to specify the public LanguageModel API."""

    def __init__(self):
        super().__init__(model="test/echo", cache=False)
        self.requests = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        return dspy.LMResponse.from_text(
            "Hello!",
            model=request.model,
            usage=dspy.LMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=0.00001,
        )


class ToolCallingLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="test/tools", cache=False)
        self.requests = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        return dspy.LMResponse(
            model=request.model,
            outputs=[
                dspy.lm.LMOutput(
                    parts=[
                        dspy.lm.LMThinkingPart(text="The calls are independent, so I can request them in parallel."),
                        dspy.lm.LMToolCallPart(
                            id="call_sf",
                            name="get_current_weather",
                            args={"location": "San Francisco, CA"},
                        ),
                        dspy.lm.LMToolCallPart(
                            id="call_tokyo",
                            name="get_current_weather",
                            args={"location": "Tokyo, Japan"},
                        ),
                        dspy.lm.LMToolCallPart(
                            id="call_paris",
                            name="get_current_weather",
                            args={"location": "Paris, France"},
                        ),
                    ],
                    finish_reason="tool_calls",
                )
            ],
            usage=dspy.LMUsage(input_tokens=20, output_tokens=30, total_tokens=50),
            cost=0.001,
        )


class RichInputLM(EchoLM):
    pass


def test_simple_call_returns_list_like_lm_response_with_metadata():
    lm = EchoLM()

    response = lm("hello")

    assert isinstance(response, dspy.LMResponse)
    assert response[0] == "Hello!"
    assert list(response) == ["Hello!"]
    assert response.text == "Hello!"
    assert response.outputs[0].text == "Hello!"
    assert response.usage.total_tokens == 2
    assert response.cost == 0.00001
    assert response.cache_hit is False
    assert response.to_legacy_outputs() == ["Hello!"]


def test_simple_call_normalizes_to_lm_request():
    lm = EchoLM()

    lm("hello")

    request = lm.requests[-1]
    assert isinstance(request, dspy.LMRequest)
    assert request.model == "test/echo"
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].parts == [dspy.lm.LMTextPart(text="hello")]


def test_common_generation_kwargs_normalize_to_lm_config():
    lm = EchoLM()

    lm("hello", temperature=0.7, max_tokens=100, top_p=0.9, stop=["END"], logprobs=True)

    config = lm.requests[-1].config
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 0.9
    assert config.stop == ["END"]
    assert config.logprobs is True


def test_openai_messages_are_boundary_input_not_internal_contract():
    lm = EchoLM()

    lm(
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "What is DSPy?"},
        ]
    )

    request = lm.requests[-1]
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].parts == [dspy.lm.LMTextPart(text="Be terse.")]
    assert request.messages[1].parts == [dspy.lm.LMTextPart(text="What is DSPy?")]


def test_message_constructors_create_multiturn_lm_messages():
    lm = EchoLM()

    lm(
        dspy.System("Be terse."),
        dspy.User("What is DSPy?"),
        dspy.Assistant("DSPy is a framework for programming LM pipelines."),
        dspy.User("Say that in five words."),
    )

    request = lm.requests[-1]
    assert [message.role for message in request.messages] == ["system", "user", "assistant", "user"]
    assert request.messages[-1].parts == [dspy.lm.LMTextPart(text="Say that in five words.")]


def test_multimodal_positional_parts_normalize_into_one_user_message():
    lm = RichInputLM()
    image = dspy.Image("data:image/png;base64,abc123")

    lm("describe this image", image)

    request = lm.requests[-1]
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].parts == [
        dspy.lm.LMTextPart(text="describe this image"),
        dspy.lm.LMImagePart(data="abc123", media_type="image/png"),
    ]


def test_explicit_lm_request_can_be_called_and_kwargs_override_config():
    lm = EchoLM()
    request = dspy.LMRequest(
        model="test/explicit",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(temperature=0.1),
    )

    lm(request, temperature=0.9)

    normalized = lm.requests[-1]
    assert normalized.model == "test/explicit"
    assert normalized.config.temperature == 0.9


def test_python_tools_normalize_to_lm_tool_specs():
    def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get the current weather in a given location."""
        return "sunny"

    lm = ToolCallingLM()

    lm(
        dspy.User("What's the weather in San Francisco, Tokyo, and Paris?"),
        tools=[dspy.Tool(get_current_weather)],
        tool_choice="required",
        parallel_tool_calls=True,
        reasoning_effort="high",
    )

    request = lm.requests[-1]
    assert len(request.tools) == 1
    assert request.tools[0] == dspy.lm.LMToolSpec(
        name="get_current_weather",
        description="Get the current weather in a given location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "default": "fahrenheit"},
            },
            "required": ["location", "unit"],
        },
    )
    assert request.config.tool_choice.mode == "required"
    assert request.config.tool_choice.parallel is True
    assert request.config.reasoning.effort == "high"


def test_tool_call_response_exposes_normalized_parts_and_legacy_view():
    lm = ToolCallingLM()

    response = lm(dspy.User("What's the weather in San Francisco, Tokyo, and Paris?"))

    assert response.text is None
    assert response.reasoning_content.startswith("The calls are independent")
    assert [call.name for call in response.tool_calls] == [
        "get_current_weather",
        "get_current_weather",
        "get_current_weather",
    ]
    assert response.tool_calls[0].args == {"location": "San Francisco, CA"}
    assert response.to_legacy_outputs() == [
        {
            "text": None,
            "reasoning_content": "The calls are independent, so I can request them in parallel.",
            "tool_calls": [
                {
                    "id": "call_sf",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "San Francisco, CA"}',
                    },
                },
                {
                    "id": "call_tokyo",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "Tokyo, Japan"}',
                    },
                },
                {
                    "id": "call_paris",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "Paris, France"}',
                    },
                },
            ],
        }
    ]


def test_tool_results_can_be_sent_back_as_normalized_tool_messages():
    lm = RichInputLM()

    lm(
        dspy.User("What's the weather in Paris?"),
        dspy.Assistant(
            dspy.LMToolCall(id="call_paris", name="get_current_weather", args={"location": "Paris, France"})
        ),
        dspy.ToolResult(
            call_id="call_paris",
            name="get_current_weather",
            content='{"location": "Paris", "temperature": "22", "unit": "celsius"}',
        ),
        dspy.User("Summarize the answer."),
    )

    request = lm.requests[-1]
    assert [message.role for message in request.messages] == ["user", "assistant", "tool", "user"]
    assert request.messages[1].parts == [
        dspy.lm.LMToolCallPart(
            id="call_paris",
            name="get_current_weather",
            args={"location": "Paris, France"},
        )
    ]
    assert request.messages[2].parts == [
        dspy.lm.LMToolResultPart(
            call_id="call_paris",
            name="get_current_weather",
            content=[dspy.lm.LMTextPart(text='{"location": "Paris", "temperature": "22", "unit": "celsius"}')],
        )
    ]


def test_history_messages_project_tool_calls_to_openai_chat_shape():
    lm = EchoLM()

    lm(
        dspy.User("What's the weather in Paris?"),
        dspy.Assistant(dspy.LMToolCall(id="call_paris", name="get_weather", args={"location": "Paris"})),
        dspy.ToolResult('{"temperature": "22"}', call_id="call_paris", name="get_weather"),
        dspy.User("Summarize."),
    )

    assert lm.history[-1]["messages"] == [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_paris",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": "22"}',
            "tool_call_id": "call_paris",
            "name": "get_weather",
        },
        {"role": "user", "content": "Summarize."},
    ]


def test_history_stores_normalized_request_response_and_compatibility_keys():
    lm = EchoLM()

    response = lm("hello")

    entry = lm.history[-1]
    assert entry.request == lm.requests[-1]
    assert entry.response == response
    assert entry["outputs"] == ["Hello!"]
    assert entry["usage"] == {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "prompt_tokens": 1, "completion_tokens": 1, "details": {}}
    assert entry["cost"] == 0.00001
    assert entry["prompt"] == "hello"
    assert entry["messages"] == [{"role": "user", "content": "hello"}]
    assert entry["model"] == "test/echo"
    assert "timestamp" in entry
    assert "uuid" in entry


def test_cache_hit_response_does_not_count_new_usage_or_cost():
    response = dspy.LMResponse.from_text(
        "cached",
        model="test/cache",
        cache_hit=True,
        usage=None,
        cost=None,
    )

    assert response.cache_hit is True
    assert response.usage is None
    assert response.cost is None
    assert response.usage_as_dict() == {}
