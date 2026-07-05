"""Tests for benchmark smoke experiment model configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import dspy
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
QWEN3_4B_MODEL_CONFIG = CONFIGS_DIR / "models" / "qwen3_4b.yaml"
SMOKE_EXPERIMENTS = ("ifeval_baseline_smoke", "ifeval_sbo_smoke")


def _load_model_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ollama_reachable() -> bool:
    try:
        import urllib.error
        import urllib.request

        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except (OSError, urllib.error.URLError):
        return False


def test_qwen3_4b_model_config_is_instruct() -> None:
    """Smoke benchmarks should use the non-thinking instruct model."""
    model_config = _load_model_yaml(QWEN3_4B_MODEL_CONFIG)
    assert model_config["name"] == "ollama_chat/qwen3:4b-instruct"
    assert "qwen3.5" not in model_config["name"]


@pytest.mark.parametrize("config_name", SMOKE_EXPERIMENTS)
def test_smoke_experiments_use_qwen3_4b_instruct(config_name: str) -> None:
    config = load_config(CONFIGS_DIR / "experiments" / f"{config_name}.yaml")
    assert config.model.name == "ollama_chat/qwen3:4b-instruct"


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running on localhost:11434")
def test_qwen3_4b_returns_parsed_response() -> None:
    """Live check: default instruct model must produce a non-None parsed response."""
    model_config = _load_model_yaml(QWEN3_4B_MODEL_CONFIG)
    lm = dspy.LM(
        model=model_config["name"],
        api_base=model_config.get("api_base", "http://localhost:11434"),
        cache=False,
        **model_config.get("params", {}),
    )
    dspy.configure(lm=lm)

    prediction = dspy.Predict("prompt -> response")(prompt="Reply with exactly one word: hello")

    assert getattr(prediction, "response", None) is not None
    assert str(prediction.response).strip()
