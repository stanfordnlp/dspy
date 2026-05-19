from unittest import mock

import pytest

import dspy


class AdapterEchoLM(dspy.BaseLM):
    def __init__(self, *, response=None, **kwargs):
        super().__init__(model="test/adapter-echo", cache=False, **kwargs)
        self.response = response or dspy.LMResponse.from_text("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]", model=self.model)
        self.requests = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        return self.response

    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        return self.response


def test_adapter_accepts_normalized_language_model_outputs():
    signature = dspy.make_signature("question -> answer")
    adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
    lm = AdapterEchoLM()

    result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

    assert result == [{"answer": "Paris"}]
    assert isinstance(lm.requests[0], dspy.LMRequest)


@pytest.mark.asyncio
async def test_two_step_adapter_async_uses_language_model_branch_for_main_and_extraction_models():
    signature = dspy.make_signature("question -> answer")
    main_lm = AdapterEchoLM(response=dspy.LMResponse.from_text("The answer is Paris.", model="test/main"))
    extraction_lm = AdapterEchoLM(
        response=dspy.LMResponse.from_text("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]", model="test/extract")
    )
    adapter = dspy.TwoStepAdapter(extraction_model=extraction_lm)

    result = await adapter.acall(main_lm, {}, signature, [], {"question": "Capital?"})

    assert result == [{"answer": "Paris"}]
    assert isinstance(main_lm.requests[0], dspy.LMRequest)
    assert isinstance(extraction_lm.requests[0], dspy.LMRequest)


@pytest.mark.asyncio
async def test_two_step_adapter_async_preserves_normalized_tool_calls():
    class MainToolLM(AdapterEchoLM):
        def __init__(self):
            super().__init__(
                response=dspy.LMResponse(
                    model="test/main-tool",
                    outputs=[
                        dspy.lm.LMOutput(
                            parts=[
                                dspy.lm.LMTextPart(text="I should call search."),
                                dspy.lm.LMToolCallPart(name="search", args={"query": "DSPy"}),
                            ]
                        )
                    ],
                )
            )

    class QAWithTools(dspy.Signature):
        question: str = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()
        answer: str = dspy.OutputField()

    extraction_lm = AdapterEchoLM(
        response=dspy.LMResponse.from_text("[[ ## answer ## ]]\nNeed search.\n\n[[ ## completed ## ]]", model="test/extract")
    )
    adapter = dspy.TwoStepAdapter(extraction_model=extraction_lm)

    result = await adapter.acall(MainToolLM(), {}, QAWithTools, [], {"question": "Find DSPy"})

    assert result[0]["answer"] == "Need search."
    assert result[0]["tool_calls"] == dspy.ToolCalls.from_dict_list(
        [{"name": "search", "args": {"query": "DSPy"}}]
    )


def test_adapter_extracts_native_reasoning_from_normalized_language_model_outputs():
    class ReasoningLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="test/reasoning", cache=False)

        def get_capabilities(self) -> dspy.LMCapabilities:
            return dspy.LMCapabilities(reasoning=True)

        def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
            assert request.config.reasoning == dspy.lm.LMReasoningConfig(effort="low")
            return dspy.LMResponse(
                model=request.model,
                outputs=[
                    dspy.lm.LMOutput(
                        parts=[
                            dspy.lm.LMThinkingPart(text="Think first."),
                            dspy.lm.LMTextPart(text="[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"),
                        ]
                    )
                ],
            )

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    result = dspy.ChatAdapter()(ReasoningLM(), {}, QA, [], {"question": "Capital?"})

    assert result == [{"answer": "Paris", "reasoning": dspy.Reasoning("Think first.")}]


def test_xml_adapter_accepts_normalized_language_model_outputs():
    signature = dspy.make_signature("question -> answer")
    adapter = dspy.XMLAdapter()
    lm = AdapterEchoLM(response=dspy.LMResponse.from_text("<answer>Paris</answer>", model="test/adapter-echo"))

    result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

    assert result == [{"answer": "Paris"}]
    assert isinstance(lm.requests[0], dspy.LMRequest)


def test_baml_adapter_accepts_normalized_language_model_outputs():
    from dspy.adapters.baml_adapter import BAMLAdapter

    signature = dspy.make_signature("question -> answer")
    adapter = BAMLAdapter()
    lm = AdapterEchoLM(response=dspy.LMResponse.from_text('{"answer": "Paris"}', model="test/adapter-echo"))

    result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

    assert result == [{"answer": "Paris"}]
    assert isinstance(lm.requests[0], dspy.LMRequest)


def test_adapter_language_model_branch_preserves_constructor_defaults():
    class DefaultConfigLM(AdapterEchoLM):
        def __init__(self):
            super().__init__(temperature=0.2, max_tokens=123, service_tier="auto")

    signature = dspy.make_signature("question -> answer")
    lm = DefaultConfigLM()

    dspy.ChatAdapter()(lm, {}, signature, [], {"question": "What is the capital of France?"})

    request = lm.requests[0]
    assert request.config.temperature == 0.2
    assert request.config.max_tokens == 123
    assert request.config.extensions["service_tier"] == "auto"


def test_json_adapter_uses_language_model_capabilities_not_legacy_properties():
    signature = dspy.make_signature("question -> answer")
    adapter = dspy.JSONAdapter()
    lm = dspy.OpenAIChatLM("openai/gpt-4o-mini", cache=False)

    assert lm.capabilities.function_calling is True
    assert lm.capabilities.reasoning is True
    assert lm.capabilities.response_schema is True
    # BaseLM unification: every LM exposes the legacy property surface; the
    # backing values come from `lm.capabilities`, not LiteLLM probing.
    assert lm.supported_params == set()

    with mock.patch.object(lm, "forward", return_value=dspy.LMResponse.from_text('{"answer": "Paris"}', model=lm.model)) as forward:
        result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

    assert result == [{"answer": "Paris"}]
    assert forward.call_args.args[0].config.response_format is not None


def test_router_accepts_api_base_and_num_retries():
    lm = dspy.LMRouter(
        "openai/gpt-4o-mini",
        api_base="https://example.test/v1",
        api_key="secret",
        cache=False,
        num_retries=2,
    )

    assert isinstance(lm, dspy.OpenAIResponsesLM)
    assert lm.api_base == "https://example.test/v1"
    assert lm.api_key == "secret"
    assert lm.num_retries == 2


def test_router_rejects_legacy_only_lm_kwargs_instead_of_forwarding_them():
    with pytest.raises(TypeError, match=r"legacy `dspy[.]LM` constructor arguments"):
        dspy.LMRouter("openai/gpt-4o-mini", provider=object())


def test_prefix_registration_accepts_language_model_subclasses():
    class AcmeLM(dspy.BaseLM):
        def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
            return dspy.LMResponse.from_text("acme", model=request.model)

    dspy.register_lm_backend(AcmeLM, prefix="acme-test")

    lm = dspy.LMRouter("acme-test/small", cache=False)

    assert isinstance(lm, AcmeLM)
    assert lm("hello").text == "acme"
