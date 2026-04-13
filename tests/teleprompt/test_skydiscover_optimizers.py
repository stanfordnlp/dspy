"""
Tests for skydiscover-based DSPy optimizers (AdaEvolve, EvoX).

Uses a simple sentiment classification task with a mock LM to verify
the optimization loop works end-to-end without requiring API keys.
"""

import dspy
import pytest


# ---------------------------------------------------------------------------
# Fixtures: tiny sentiment classification task
# ---------------------------------------------------------------------------

SENTIMENT_DATA = [
    ("I love this product, it's amazing!", "positive"),
    ("Terrible experience, would not recommend.", "negative"),
    ("Best purchase I ever made.", "positive"),
    ("Awful quality, broke after one day.", "negative"),
    ("Absolutely wonderful, exceeded expectations!", "positive"),
    ("Waste of money, very disappointed.", "negative"),
    ("Great value for the price.", "positive"),
    ("Horrible customer service.", "negative"),
]


def make_examples(data):
    return [
        dspy.Example(text=text, sentiment=label).with_inputs("text")
        for text, label in data
    ]


class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("text -> sentiment")

    def forward(self, text):
        return self.classify(text=text)


def sentiment_metric(gold, pred, trace=None):
    """Simple exact-match metric."""
    pred_val = getattr(pred, "sentiment", "").strip().lower()
    gold_val = gold.sentiment.strip().lower()
    return float(pred_val == gold_val)


# ---------------------------------------------------------------------------
# Mock LM that returns sentiment based on simple keyword matching,
# and handles instruction proposal / prompt improvement prompts
# ---------------------------------------------------------------------------

class MockSentimentLM(dspy.LM):
    """
    A mock LM that:
    - For sentiment tasks: returns positive/negative based on keywords
    - For prompt improvement (skydiscover context builder): returns improved instructions
    - For search strategy generation: returns mock strategy code
    """

    def __init__(self):
        self.provider = "mock"
        self.model = "mock-sentiment"
        self.model_type = "chat"
        self.cache = False
        self.history = []
        self.kwargs = {}
        self._call_count = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        self._call_count += 1

        # Get the text to analyze
        text = ""
        if isinstance(prompt, str):
            text = prompt
        elif messages:
            for m in messages:
                if isinstance(m, dict):
                    text += m.get("content", "")
                elif isinstance(m, str):
                    text += m

        text_lower = text.lower()

        # Check if this is a prompt improvement request from the context builder
        # (skydiscover's default/adaevolve templates ask to improve prompts)
        if "suggest improvements" in text_lower or "your rewritten prompt" in text_lower:
            return [
                '```text\n{"classify": "Classify the sentiment of the given text as either '
                "'positive' or 'negative'. Look for emotional keywords: "
                "words like 'love', 'great', 'amazing', 'wonderful' indicate positive; "
                "words like 'terrible', 'awful', 'horrible', 'waste' indicate negative. "
                'Return exactly one word."}\n```'
            ]

        # Check if this is an instruction proposal request (GEPA-style)
        if "write a new instruction" in text_lower or "current_instruction_doc" in text_lower:
            return [
                "```\nClassify the sentiment of the given text as either "
                "'positive' or 'negative'. Look for emotional keywords.\n```"
            ]

        # Check if this is a search strategy generation request
        if "EvolvedProgramDatabase" in text or "evolvedprogramdatabase" in text_lower:
            return [MOCK_STRATEGY_CODE]

        # Sentiment classification
        positive_words = ["love", "great", "best", "wonderful", "amazing", "exceeded"]
        negative_words = ["terrible", "awful", "horrible", "waste", "broke", "disappointed"]

        for word in positive_words:
            if word in text_lower:
                return ["positive"]
        for word in negative_words:
            if word in text_lower:
                return ["negative"]

        return ["positive"]

    def inspect_history(self, n=1):
        return self.history[-n:]


# Mock strategy code returned by the "guide LM" for EvoX
MOCK_STRATEGY_CODE = """\
# EVOLVE-BLOCK-START
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase


@dataclass
class EvolvedProgram(Program):
    pass


class EvolvedProgramDatabase(ProgramDatabase):
    def __init__(self, name, config):
        super().__init__(name, config)

    def add(self, program, iteration=None, **kwargs):
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        self._update_best_program(program)
        return program.id

    def sample(self, num_context_programs=4, **kwargs):
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates")
        if len(candidates) >= 2:
            a, b = random.sample(candidates, 2)
            score_a = a.metrics.get("combined_score", 0) if a.metrics else 0
            score_b = b.metrics.get("combined_score", 0) if b.metrics else 0
            parent = a if score_a >= score_b else b
        else:
            parent = candidates[0]
        others = [p for p in candidates if p.id != parent.id]
        context = random.sample(others, min(num_context_programs, len(others)))
        return {"": parent}, {"": context}
# EVOLVE-BLOCK-END
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_lm():
    lm = MockSentimentLM()
    return lm


@pytest.fixture
def trainset():
    return make_examples(SENTIMENT_DATA[:6])


@pytest.fixture
def valset():
    return make_examples(SENTIMENT_DATA[6:])


class TestAdaEvolve:
    def test_basic_compile(self, mock_lm, trainset, valset):
        """AdaEvolve.compile() runs end-to-end and returns a compiled Module."""
        dspy.configure(lm=mock_lm)

        optimizer = dspy.AdaEvolve(
            metric=sentiment_metric,
            reflection_lm=mock_lm,
            max_iterations=3,
            num_islands=1,
            population_size=5,
            seed=42,
        )

        student = SentimentClassifier()
        optimized = optimizer.compile(student, trainset=trainset, valset=valset)

        assert isinstance(optimized, dspy.Module)
        assert optimized._compiled is True

        preds = dict(optimized.named_predictors())
        assert "classify" in preds

    def test_trainset_as_valset(self, mock_lm, trainset):
        """When no valset provided, trainset is used for both."""
        dspy.configure(lm=mock_lm)

        optimizer = dspy.AdaEvolve(
            metric=sentiment_metric,
            reflection_lm=mock_lm,
            max_iterations=2,
            num_islands=1,
            population_size=5,
            seed=0,
        )

        optimized = optimizer.compile(SentimentClassifier(), trainset=trainset)
        assert isinstance(optimized, dspy.Module)

    def test_multiple_islands(self, mock_lm, trainset, valset):
        """Multiple islands run without error."""
        dspy.configure(lm=mock_lm)

        optimizer = dspy.AdaEvolve(
            metric=sentiment_metric,
            reflection_lm=mock_lm,
            max_iterations=4,
            num_islands=2,
            population_size=5,
            seed=42,
        )

        optimized = optimizer.compile(
            SentimentClassifier(), trainset=trainset, valset=valset,
        )
        assert isinstance(optimized, dspy.Module)


class TestEvoX:
    def test_basic_compile(self, mock_lm, trainset, valset):
        """EvoX.compile() runs end-to-end and returns a compiled Module."""
        dspy.configure(lm=mock_lm)

        optimizer = dspy.EvoX(
            metric=sentiment_metric,
            reflection_lm=mock_lm,
            guide_lm=mock_lm,
            max_iterations=3,
            seed=42,
        )

        student = SentimentClassifier()
        optimized = optimizer.compile(student, trainset=trainset, valset=valset)

        assert isinstance(optimized, dspy.Module)
        assert optimized._compiled is True

    def test_trainset_as_valset(self, mock_lm, trainset):
        """When no valset provided, trainset is used for both."""
        dspy.configure(lm=mock_lm)

        optimizer = dspy.EvoX(
            metric=sentiment_metric,
            reflection_lm=mock_lm,
            max_iterations=2,
            seed=0,
        )

        optimized = optimizer.compile(SentimentClassifier(), trainset=trainset)
        assert isinstance(optimized, dspy.Module)

    def test_strategy_evolution_triggered(self, mock_lm, trainset, valset):
        """With low switch_ratio and enough iterations, strategy evolution triggers."""
        dspy.configure(lm=mock_lm)

        optimizer = dspy.EvoX(
            metric=sentiment_metric,
            reflection_lm=mock_lm,
            guide_lm=mock_lm,
            max_iterations=6,
            switch_ratio=0.3,
            improvement_threshold=10.0,
            seed=42,
        )

        optimized = optimizer.compile(
            SentimentClassifier(), trainset=trainset, valset=valset,
        )
        assert isinstance(optimized, dspy.Module)


class TestAdapter:
    """Test shared adapter utilities."""

    def test_candidate_serialization(self):
        """Candidate dict round-trips through JSON serialization."""
        from dspy.teleprompt.skydiscover.adapter import (
            candidate_to_solution,
            solution_to_candidate,
        )

        candidate = {"pred_a": "Instruction A", "pred_b": "Instruction B"}
        solution = candidate_to_solution(candidate)
        recovered = solution_to_candidate(solution)
        assert recovered == candidate

    def test_build_dspy_program(self, mock_lm):
        """build_dspy_program injects instructions correctly."""
        from dspy.teleprompt.skydiscover.adapter import build_dspy_program

        dspy.configure(lm=mock_lm)
        student = SentimentClassifier()
        original = student.classify.signature.instructions

        candidate = {"classify": "New custom instruction for classification"}
        built = build_dspy_program(student, candidate)

        # Original should be unchanged
        assert student.classify.signature.instructions == original
        # Built should have new instruction
        assert built.classify.signature.instructions == "New custom instruction for classification"

    def test_extract_seed_candidate(self, mock_lm):
        """extract_seed_candidate pulls current instructions."""
        from dspy.teleprompt.skydiscover.adapter import extract_seed_candidate

        dspy.configure(lm=mock_lm)
        student = SentimentClassifier()
        candidate = extract_seed_candidate(student)

        assert "classify" in candidate
        assert isinstance(candidate["classify"], str)
        assert len(candidate["classify"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
