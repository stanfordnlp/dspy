import math
import random

import pytest

import dspy
from dspy.utils.dummies import DummyLM


@pytest.mark.parametrize(
    "score, sampling_temperature, candidate_temperature",
    [(1.0, 0.001, 0.2), (1.0, 0.2, 0.001), (1000.0, 0.2, 0.2), (0.0, 0.2, 0.2), (-1000.0, 0.2, 0.2)],
)
def test_simba_sampling_remains_finite(score, sampling_temperature, candidate_temperature):
    dspy.configure(lm=DummyLM({"q": {"answer": "ok"}}))
    optimizer = dspy.SIMBA(
        metric=lambda example, prediction: score,
        bsize=1,
        num_candidates=2,
        max_steps=2,
        num_threads=1,
        temperature_for_sampling=sampling_temperature,
        temperature_for_candidates=candidate_temperature,
    )

    optimized = optimizer.compile(
        dspy.Predict("question -> answer"),
        trainset=[dspy.Example(question="q", answer="ok").with_inputs("question")],
    )

    assert optimized.candidate_programs
    assert all(candidate["score"] == score for candidate in optimized.candidate_programs)


@pytest.mark.parametrize("temperature", [0.2, 0.1, 0.001, 0.0001])
def test_simba_softmax_preserves_relative_weights(monkeypatch, temperature):
    """Sampling must retain the softmax probabilities at each temperature."""
    recorded_weights = []
    original_choices = random.Random.choices

    def record_choices(self, population, weights=None, **kwargs):
        if len(population) > 1:
            recorded_weights.append(list(weights))
        return original_choices(self, population, weights=weights, **kwargs)

    monkeypatch.setattr(random.Random, "choices", record_choices)

    class ScoredProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.score = 0.0
            self.predictor = dspy.Predict("question -> answer")

        def forward(self, question):
            return dspy.Prediction(score=self.score)

    def improve(bucket, program, **kwargs):
        program.score += 0.25
        return True

    dspy.configure(lm=DummyLM([]))
    optimizer = dspy.SIMBA(
        metric=lambda example, prediction: prediction.score,
        bsize=1,
        num_candidates=2,
        max_steps=2,
        num_threads=1,
        temperature_for_sampling=temperature,
        temperature_for_candidates=temperature,
    )
    optimizer.strategies = [improve]
    optimizer.compile(
        ScoredProgram(),
        trainset=[dspy.Example(question="q").with_inputs("question")],
    )

    # At the start of step two the sampler sees scores 0.0 and 0.25.
    expected_best_probability = 1 / (1 + math.exp(-0.25 / temperature))
    assert recorded_weights
    assert recorded_weights[0] == pytest.approx(
        [expected_best_probability, 1 - expected_best_probability]
    )
