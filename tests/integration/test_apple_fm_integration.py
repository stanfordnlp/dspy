"""Integration tests for the Apple Foundation Models DSPy adapter.

These tests hit the real on-device model and MUST run on macOS 26+ with
Apple Intelligence enabled.  They are skipped automatically on any other
platform (Linux CI, WSL, older macOS).

Run on Mac with::

    pytest tests/integration/test_apple_fm_integration.py -v

Fixture recording
-----------------
The ``record_response`` fixture (defined in conftest.py) optionally captures
real request/response pairs to ``tests/fixtures/apple_fm_recordings/`` so
the unit tests can replay them as authoritative mocks.

To record a fresh set of fixtures::

    APPLE_FM_RECORD=1 pytest tests/integration/ -v
"""

from __future__ import annotations

import json
import platform
from typing import Literal

import pytest
from pydantic import BaseModel, Field

if platform.system() != "Darwin":
    pytest.skip("Requires macOS 26+ with Apple Intelligence enabled", allow_module_level=True)

pytest.importorskip("apple_fm_sdk", reason="apple-fm-sdk is not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lm():
    """Return a configured AppleFoundationLM for the test session."""
    import dspy

    _lm = dspy.AppleFoundationLM(cache=False)
    dspy.configure(lm=_lm)
    return _lm


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestBasicGeneration:
    def test_plain_text_response(self, lm):
        result = lm.forward(prompt="Reply with exactly the word: hello")
        text = result.choices[0].message.content
        assert isinstance(text, str)
        assert len(text) > 0

    def test_messages_roundtrip(self, lm):
        result = lm.forward(messages=[
            {"role": "system", "content": "Answer questions clearly and concisely."},
            {"role": "user", "content": "What is the capital of France?"},
        ])
        text = result.choices[0].message.content.strip()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_response_model_field(self, lm):
        result = lm.forward(prompt="hi")
        assert result.model == "apple/on-device"

    def test_usage_dict_serialisable(self, lm):
        result = lm.forward(prompt="hi")
        usage = dict(result.usage)
        assert set(usage.keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    def test_dspy_predict_roundtrip(self, lm):
        import dspy

        # Re-configure each time: conftest.py's autouse clear_settings fixture resets
        # dspy.configure() after every test, so the module-scoped fixture's configure call
        # does not persist to subsequent tests.
        dspy.configure(lm=lm)

        pred = dspy.Predict("question -> answer")
        output = pred(question="What colour is the sky on a clear day?")
        assert isinstance(output.answer, str)
        assert len(output.answer) > 0


# ---------------------------------------------------------------------------
# Native guided generation
# ---------------------------------------------------------------------------


class TestGuidedGeneration:
    def test_literal_field_constrained(self, lm):
        class Sentiment(BaseModel):
            label: Literal["positive", "negative", "neutral"]

        result = lm.forward(
            prompt="Classify this review: 'I love this product!'",
            response_format=Sentiment,
        )
        parsed = json.loads(result.choices[0].message.content)
        assert parsed["label"] in {"positive", "negative", "neutral"}

    def test_int_range_field(self, lm):
        class Rating(BaseModel):
            score: int = Field(ge=1, le=5)

        result = lm.forward(
            prompt="Rate the following on a scale of 1-5: 'Outstanding quality!'",
            response_format=Rating,
        )
        parsed = json.loads(result.choices[0].message.content)
        assert 1 <= parsed["score"] <= 5

    def test_multi_field_schema(self, lm):
        class Review(BaseModel):
            sentiment: Literal["positive", "negative", "neutral"]
            score: int = Field(ge=1, le=5)
            summary: str

        result = lm.forward(
            prompt=(
                "Analyse this product review and return structured data: "
                "'Absolutely fantastic build quality, would buy again!'"
            ),
            response_format=Review,
        )
        parsed = json.loads(result.choices[0].message.content)
        assert parsed["sentiment"] in {"positive", "negative", "neutral"}
        assert 1 <= parsed["score"] <= 5
        assert isinstance(parsed["summary"], str)


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


class TestToolCalling:
    def test_tool_is_called(self, lm):
        tool_calls: list[dict] = []

        def add(a: int, b: int) -> str:
            tool_calls.append({"a": a, "b": b})
            return str(a + b)

        add.name = "add"

        result = lm.forward(
            prompt="Use the add tool to compute 3 + 7",
            tools=[add],
        )
        # Either the tool was invoked during generation, or the response mentions the sum
        text = result.choices[0].message.content
        assert tool_calls or "10" in text


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestAsync:
    def test_aforward(self, lm):
        import asyncio

        async def _run():
            return await lm.aforward(prompt="Say: async works")

        result = asyncio.run(_run())
        text = result.choices[0].message.content
        assert isinstance(text, str) and len(text) > 0


# ---------------------------------------------------------------------------
# DSPy optimiser smoke test (capability check, not correctness)
# ---------------------------------------------------------------------------


class TestOptimiserSmoke:
    def test_bootstrap_few_shot_compiles(self, lm):
        """Verify BootstrapFewShot can compile a simple module against the on-device model.

        This is a smoke test only — we're checking that the adapter doesn't
        crash the optimiser, not that it produces a great few-shot prompt.
        """
        import dspy
        from dspy.teleprompt import BootstrapFewShot

        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        trainset = [
            dspy.Example(question="What is 1+1?", answer="2").with_inputs("question"),
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        ]

        def exact_match(example, pred, trace=None):
            return example.answer.strip() == pred.answer.strip()

        teleprompter = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=1)
        compiled = teleprompter.compile(dspy.Predict(QA), trainset=trainset)
        assert compiled is not None
