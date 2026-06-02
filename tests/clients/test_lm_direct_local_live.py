"""Live local-backend coverage for the experimental direct LM call interface.

These tests are opt-in because they require local model runtimes:

- `pytest tests/clients/test_lm_direct_local_live.py --vllm`
- `pytest tests/clients/test_lm_direct_local_live.py --ollama`

The vLLM test loads an offline in-process model. The Ollama test expects a running Ollama daemon with the requested
model already pulled.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import requests

import dspy

VLLM_MODEL_ENV = "DSPY_TEST_VLLM_MODEL"
OLLAMA_MODEL_ENV = "DSPY_TEST_OLLAMA_MODEL"
OLLAMA_BASE_URL_ENV = "DSPY_TEST_OLLAMA_BASE_URL"


def _assert_nonempty_lm_text(response: dspy.LMResponse) -> str:
    assert isinstance(response, dspy.LMResponse)
    assert response.text is not None
    text = response.text.strip()
    assert text
    return text


@pytest.mark.vllm
def test_live_vllm_typed_baselm_direct_messages():
    """Run the typed BaseLM contract against an in-process local vLLM model."""

    vllm = pytest.importorskip("vllm")

    class VLLMLM(dspy.BaseLM):
        forward_contract = "typed_lm"

        def __init__(self, model: str, **kwargs: Any):
            super().__init__(model=model, **kwargs)
            self.llm = vllm.LLM(
                model=model,
                trust_remote_code=os.getenv("DSPY_TEST_VLLM_TRUST_REMOTE_CODE", "false").lower() == "true",
            )
            self.tokenizer = self.llm.get_tokenizer()

        def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
            messages = [{"role": message.role, "content": message.text or ""} for message in request.messages]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            params = vllm.SamplingParams(
                temperature=request.config.temperature if request.config.temperature is not None else 0.0,
                max_tokens=request.config.max_tokens or 32,
                stop=request.config.stop,
            )
            output = self.llm.generate([prompt], params)[0]
            return dspy.LMResponse.from_text(output.outputs[0].text, model=request.model)

    lm = VLLMLM(
        model=os.getenv(VLLM_MODEL_ENV, "Qwen/Qwen2.5-0.5B-Instruct"),
        temperature=0.0,
        max_tokens=32,
        cache=False,
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the user's requested exact final token. No punctuation."),
            dspy.User("Reply with exactly: beta"),
        )

    assert "beta" in _assert_nonempty_lm_text(response).lower()


@pytest.mark.ollama
def test_live_ollama_typed_baselm_direct_messages():
    """Run the typed BaseLM contract against a local Ollama daemon."""

    base_url = os.getenv(OLLAMA_BASE_URL_ENV, "http://localhost:11434").rstrip("/")
    model = os.getenv(OLLAMA_MODEL_ENV, "llama3.2")

    try:
        tags_response = requests.get(f"{base_url}/api/tags", timeout=5)
        tags_response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Ollama is not reachable at {base_url}: {exc}")

    models = tags_response.json().get("models", [])
    model_names = {item.get("name") for item in models if item.get("name")}
    if model not in model_names:
        base_matches = sorted(name for name in model_names if name.split(":", 1)[0] == model.split(":", 1)[0])
        if not base_matches:
            pytest.skip(f"Ollama model {model!r} is not pulled. Run `ollama pull {model}` first.")
        # Ollama's generate/chat endpoints require the exact local tag even if
        # `/api/tags` shows an obvious base-name match like `llama3.2:1b`.
        model = base_matches[0]

    class OllamaLM(dspy.BaseLM):
        forward_contract = "typed_lm"

        def __init__(self, model: str, base_url: str, **kwargs: Any):
            super().__init__(model=model, **kwargs)
            self.base_url = base_url

        def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
            messages = [{"role": message.role, "content": message.text or ""} for message in request.messages]
            options = {
                "temperature": request.config.temperature if request.config.temperature is not None else 0.0,
                "num_predict": request.config.max_tokens or 32,
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False, "options": options},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return dspy.LMResponse.from_text(data.get("message", {}).get("content", ""), model=request.model)

    lm = OllamaLM(model=model, base_url=base_url, temperature=0.0, max_tokens=32, cache=False)

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the user's requested exact final token. No punctuation."),
            dspy.User("Reply with exactly: beta"),
        )

    assert "beta" in _assert_nonempty_lm_text(response).lower()
