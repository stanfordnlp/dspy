import dspy


def test_router_auto_routes_groq_chat_models(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret")

    with dspy.context(experimental=True):
        lm = dspy.LM("groq/llama-3.3-70b-versatile", cache=False)

    assert isinstance(lm, dspy.OpenAIChatLM)
    assert lm.model == "llama-3.3-70b-versatile"
    assert lm.api_base == "https://api.groq.com/openai/v1"
    assert lm.api_key == "secret"


def test_router_auto_routes_groq_gpt_oss_to_responses(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret")

    with dspy.context(experimental=True):
        lm = dspy.LM("groq/openai/gpt-oss-120b", cache=False)

    assert isinstance(lm, dspy.OpenAIResponsesLM)
    assert lm.model == "openai/gpt-oss-120b"
    assert lm.api_base == "https://api.groq.com/openai/v1"
    assert lm.api_key == "secret"


def test_router_auto_routes_ollama_to_local_openai_compatible_chat():
    with dspy.context(experimental=True):
        lm = dspy.LM("ollama/llama3.2", cache=False)

    assert isinstance(lm, dspy.OpenAIChatLM)
    assert lm.model == "llama3.2"
    assert lm.api_base == "http://localhost:11434/v1"
    assert lm.api_key == "ollama"


def test_router_infers_backend_from_complete_endpoint_url():
    with dspy.context(experimental=True):
        responses_lm = dspy.LM("custom-model", endpoint_url="https://proxy.example.test/invoke/responses", cache=False)
        chat_lm = dspy.LM(
            "custom-model",
            endpoint_url="https://proxy.example.test/invoke/chat/completions",
            cache=False,
        )

    assert isinstance(responses_lm, dspy.OpenAIResponsesLM)
    assert responses_lm.endpoint_url == "https://proxy.example.test/invoke/responses"
    assert isinstance(chat_lm, dspy.OpenAIChatLM)
    assert chat_lm.endpoint_url == "https://proxy.example.test/invoke/chat/completions"


def test_unknown_prefixed_models_fall_back_to_litellm_backend():
    with dspy.context(experimental=True):
        lm = dspy.LM("bedrock/anthropic.claude-3-sonnet", cache=False)

    assert isinstance(lm, dspy.LiteLLMLM)
    assert lm.model == "bedrock/anthropic.claude-3-sonnet"
    assert lm.model_type == "chat"


def test_router_omitted_num_retries_preserves_backend_default():
    with dspy.context(experimental=True):
        lm = dspy.LM("bedrock/anthropic.claude-3-sonnet", cache=False)

    assert isinstance(lm, dspy.LiteLLMLM)
    assert lm.num_retries == 3
