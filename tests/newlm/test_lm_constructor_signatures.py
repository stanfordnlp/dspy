import inspect

import dspy
from dspy.clients.language_models.router import LMRouter


def _parameter_names(callable_object):
    return set(inspect.signature(callable_object).parameters)


def test_public_lm_router_exposes_common_ux_kwargs():
    assert {
        "api_key",
        "api_base",
        "temperature",
        "max_tokens",
        "cache",
        "callbacks",
    } <= _parameter_names(LMRouter)


def test_language_model_base_exposes_common_generation_kwargs():
    assert {"temperature", "max_tokens", "cache", "callbacks"} <= _parameter_names(dspy.BaseLM.__init__)


def test_provider_backends_expose_common_auth_transport_and_generation_kwargs():
    required = {
        "api_key",
        "api_base",
        "temperature",
        "max_tokens",
        "cache",
        "callbacks",
    }

    for cls in [dspy.OpenAIChatLM, dspy.OpenAITextLM, dspy.OpenAIResponsesLM, dspy.AnthropicLM, dspy.GenAILM]:
        assert required <= _parameter_names(cls.__init__)


def test_explicit_constructor_kwargs_become_default_request_config():
    lm = dspy.OpenAIChatLM(
        "openai/gpt-4o-mini",
        api_key="secret",
        api_base="https://example.test/v1",
        temperature=0.2,
        max_tokens=123,
        cache=False,
    )

    request = lm.normalize_request("hello")

    assert lm.api_key == "secret"
    assert lm.api_base == "https://example.test/v1"
    assert request.config.temperature == 0.2
    assert request.config.max_tokens == 123
    assert request.config.cache.enabled is False
