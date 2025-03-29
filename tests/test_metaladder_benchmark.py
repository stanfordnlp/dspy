"""Tests for MetaLadder benchmarking functionality."""

from typing import Dict, List, Tuple
import pytest
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from examples.metaladder_benchmark import (
    BenchmarkResult,
    MathProblemSet,
    run_benchmark,
    benchmark_models
)
from examples.metaladder_vs_cot import StandardCoTModel
from examples.models.simple_math_model import SimpleMathModel
from dspy.adapters.metaladder_adapter import MetaLadderAdapter


@pytest.fixture
def problem_set() -> MathProblemSet:
    """Fixture providing a MathProblemSet instance."""
    return MathProblemSet()


@pytest.fixture
def models() -> Tuple[StandardCoTModel, MetaLadderAdapter]:
    """Fixture providing initialized models."""
    cot_model = StandardCoTModel()
    meta_model = SimpleMathModel()
    adapter = MetaLadderAdapter(model=meta_model, use_shortcut=False)
    return cot_model, adapter


def test_problem_set_structure(problem_set: MathProblemSet) -> None:
    """Test that the problem set has the expected structure."""
    # Check base problems
    assert set(problem_set.base_problems.keys()) == {"division", "percentage", "rate"}
    for problems in problem_set.base_problems.values():
        assert len(problems) == 2
        for problem, answer in problems:
            assert isinstance(problem, str)
            assert isinstance(answer, str)
    
    # Check variation problems
    assert set(problem_set.variation_problems.keys()) == {"division", "percentage", "rate"}
    for problems in problem_set.variation_problems.values():
        assert len(problems) == 1
        for problem, answer in problems:
            assert isinstance(problem, str)
            assert isinstance(answer, str)


def test_benchmark_result_creation() -> None:
    """Test BenchmarkResult creation and attributes."""
    result = BenchmarkResult(
        accuracy=85.5,
        avg_time=0.5,
        problem_types={"division": 90.0, "percentage": 80.0},
        generalization_score=75.0
    )
    
    assert result.accuracy == 85.5
    assert result.avg_time == 0.5
    assert result.problem_types == {"division": 90.0, "percentage": 80.0}
    assert result.generalization_score == 75.0


def test_run_benchmark(
    models: Tuple[StandardCoTModel, MetaLadderAdapter],
    problem_set: MathProblemSet
) -> None:
    """Test the run_benchmark function."""
    cot_model, adapter = models
    problems = problem_set.base_problems["division"]
    
    # Test CoT model
    correct, time_taken = run_benchmark(cot_model, problems)
    assert isinstance(correct, int)
    assert 0 <= correct <= len(problems)
    assert time_taken > 0
    
    # Test MetaLadder model
    correct, time_taken = run_benchmark(adapter, problems, is_metaladder=True)
    assert isinstance(correct, int)
    assert 0 <= correct <= len(problems)
    assert time_taken > 0


def test_benchmark_models_output() -> None:
    """Test the structure and validity of benchmark_models output."""
    cot_results, meta_results = benchmark_models()
    
    # Test result structure
    for results in [cot_results, meta_results]:
        assert isinstance(results, BenchmarkResult)
        assert 0 <= results.accuracy <= 100
        assert results.avg_time > 0
        assert set(results.problem_types.keys()) == {"division", "percentage", "rate"}
        assert 0 <= results.generalization_score <= 100


def test_metaladder_improvement(
    capsys: CaptureFixture[str]
) -> None:
    """Test that MetaLadder shows improvement in generalization."""
    cot_results, meta_results = benchmark_models()
    
    # MetaLadder should show better generalization
    assert meta_results.generalization_score >= cot_results.generalization_score
    
    # Check output formatting
    print_benchmark_results(cot_results, meta_results)
    captured = capsys.readouterr()
    
    assert "=== Benchmark Results ===" in captured.out
    assert "Overall Accuracy:" in captured.out
    assert "Generalization Score" in captured.out
    assert "Key Findings" in captured.out 