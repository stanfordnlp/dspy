import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import dspy


@dataclass(frozen=True)
class LiveLMSpec:
    id: str
    key_name: str
    router_model: str
    concrete_type: type[dspy.LanguageModel]
    concrete_factory: Callable[[str], dspy.LanguageModel]


def _load_env_file() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def _openai_responses(_: str) -> dspy.LanguageModel:
    return dspy.OpenAIResponsesLM(
        "openai/gpt-4.1-nano",
        api_key=os.environ.get("OPENAI_API_KEY"),
        cache=False,
        temperature=0,
        max_tokens=20,
    )


def _anthropic(_: str) -> dspy.LanguageModel:
    return dspy.AnthropicLM(
        "anthropic/claude-haiku-4-5-20251001",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        cache=False,
        temperature=0,
        max_tokens=20,
    )


def _gemini(_: str) -> dspy.LanguageModel:
    return dspy.GenAILM(
        "gemini/gemini-2.5-flash-lite",
        api_key=os.environ.get("GEMINI_API_KEY"),
        cache=False,
        temperature=0,
        max_tokens=20,
    )


def _groq_openai_chat(_: str) -> dspy.LanguageModel:
    return dspy.OpenAIChatLM(
        "llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        api_base="https://api.groq.com/openai/v1",
        cache=False,
        temperature=0,
        max_tokens=20,
    )


LIVE_LM_SPECS = [
    LiveLMSpec(
        id="openai-responses-gpt-4.1-nano",
        key_name="OPENAI_API_KEY",
        router_model="openai/gpt-4.1-nano",
        concrete_type=dspy.OpenAIResponsesLM,
        concrete_factory=_openai_responses,
    ),
    LiveLMSpec(
        id="anthropic-claude-haiku",
        key_name="ANTHROPIC_API_KEY",
        router_model="anthropic/claude-haiku-4-5-20251001",
        concrete_type=dspy.AnthropicLM,
        concrete_factory=_anthropic,
    ),
    LiveLMSpec(
        id="gemini-2.5-flash-lite",
        key_name="GEMINI_API_KEY",
        router_model="gemini/gemini-2.5-flash-lite",
        concrete_type=dspy.GenAILM,
        concrete_factory=_gemini,
    ),
    LiveLMSpec(
        id="groq-openai-chat",
        key_name="GROQ_API_KEY",
        router_model="groq/llama-3.3-70b-versatile",
        concrete_type=dspy.OpenAIChatLM,
        concrete_factory=_groq_openai_chat,
    ),
]


class LiveCapitalQA(dspy.Signature):
    """Answer with exactly one city name and no punctuation."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


@pytest.mark.parametrize("spec", LIVE_LM_SPECS, ids=[spec.id for spec in LIVE_LM_SPECS])
def test_live_router_real_call_uses_expected_concrete_backend(spec: LiveLMSpec):
    if not os.environ.get(spec.key_name):
        pytest.skip(f"Set {spec.key_name} in the environment or .env to run this live LM test.")

    with dspy.context(experimental=True):
        lm = dspy.LM(spec.router_model, cache=False, temperature=0, max_tokens=20)

    assert isinstance(lm, spec.concrete_type)

    response = lm("Reply with exactly one word: PONG")

    assert response.text is not None
    assert "pong" in response.text.lower()
    assert lm.history[-1].request.model == lm.model
    assert lm.history[-1].response.text is not None


@pytest.mark.parametrize("spec", LIVE_LM_SPECS, ids=[spec.id for spec in LIVE_LM_SPECS])
def test_live_concrete_backend_real_call(spec: LiveLMSpec):
    if not os.environ.get(spec.key_name):
        pytest.skip(f"Set {spec.key_name} in the environment or .env to run this live LM test.")

    lm = spec.concrete_factory(os.environ[spec.key_name])

    assert isinstance(lm, spec.concrete_type)

    response = lm("Reply with exactly one word: PONG")

    assert response.text is not None
    assert "pong" in response.text.lower()
    assert lm.history[-1].request.model == lm.model
    assert lm.history[-1].response.text is not None


def test_live_openai_gpt_4_1_nano_predict_program_real_call():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Set OPENAI_API_KEY in the environment or .env to run this live LM test.")

    lm = dspy.OpenAIResponsesLM(
        "openai/gpt-4.1-nano",
        api_key=os.environ.get("OPENAI_API_KEY"),
        cache=False,
        temperature=0,
        max_tokens=20,
    )
    program = dspy.Predict(LiveCapitalQA)

    with dspy.context(lm=lm):
        prediction = program(question="What is the capital of France?")

    assert prediction.answer.strip().rstrip(".") == "Paris"
    assert lm.history[-1].request.model == "openai/gpt-4.1-nano"
    assert lm.history[-1].response.text is not None
