"""Tests for the MetaLadder adapter implementation."""

from typing import Any, Dict, List
import pytest
from unittest.mock import Mock, patch, call

from dspy.adapters.metaladder_adapter import MetaLadderAdapter, MetaProblem
from dspy.teleprompt import BootstrapFewShot


@pytest.fixture
def mock_model() -> Mock:
    """Create a mock model for testing."""
    model = Mock()
    model.return_value = "Test response"
    return model


@pytest.fixture
def mock_optimizer() -> Mock:
    """Create a mock optimizer for testing."""
    optimizer = Mock(spec=BootstrapFewShot)
    optimizer.optimize.return_value = "Optimized prompt"
    return optimizer


@pytest.fixture
def adapter(mock_model: Mock) -> MetaLadderAdapter:
    """Create a MetaLadder adapter instance for testing."""
    return MetaLadderAdapter(model=mock_model)


def test_init(mock_model: Mock, mock_optimizer: Mock) -> None:
    """Test MetaLadderAdapter initialization."""
    adapter = MetaLadderAdapter(
        model=mock_model, use_shortcut=True, temperature=0.5, max_tokens=512, cache_size=100, optimizer=mock_optimizer
    )

    assert adapter.model == mock_model
    assert adapter.use_shortcut is True
    assert adapter.temperature == 0.5
    assert adapter.max_tokens == 512
    assert adapter.optimizer == mock_optimizer
    assert isinstance(adapter._meta_problems, dict)


def test_identify_problem_type(adapter: MetaLadderAdapter) -> None:
    """Test problem type identification."""
    mock_response = """Problem Type: Quadratic Equation
Solution Method: Factoring"""
    adapter.model.return_value = mock_response

    problem_type, solution_method = adapter._identify_problem_type("Solve x^2 + 5x + 6 = 0")

    assert problem_type == "Quadratic Equation"
    assert solution_method == "Factoring"


def test_generate_meta_problem(adapter: MetaLadderAdapter) -> None:
    """Test meta-problem generation."""
    mock_response = """Similar Problem: Solve y^2 + 3y + 2 = 0
Solution: Factor into (y+2)(y+1)=0, so y=-2 or y=-1"""
    adapter.model.return_value = mock_response

    meta_problem = adapter._generate_meta_problem(
        question="Solve x^2 + 5x + 6 = 0", problem_type="Quadratic Equation", solution_method="Factoring"
    )

    assert isinstance(meta_problem, MetaProblem)
    assert "y^2 + 3y + 2 = 0" in meta_problem.question
    assert "Factor into" in meta_problem.solution


def test_restate_problem(adapter: MetaLadderAdapter) -> None:
    """Test problem restatement."""
    original = "Find x if x^2 + 5x + 6 = 0"
    mock_response = "Solve the quadratic equation x^2 + 5x + 6 = 0"
    adapter.model.return_value = mock_response

    restated = adapter._restate_problem(original)
    assert restated == mock_response


def test_forward_with_shortcut(mock_model: Mock) -> None:
    """Test forward pass with shortcut inference."""
    adapter = MetaLadderAdapter(model=mock_model, use_shortcut=True)

    # Mock responses for each step
    responses = [
        "Problem Type: Quadratic\nSolution Method: Factoring",  # identify_problem_type
        "Solve the quadratic equation",  # restate_problem
        "x = -2 or x = -3",  # final solution
    ]
    mock_model.side_effect = responses

    result = adapter.forward("Solve x^2 + 5x + 6 = 0")

    assert result.text == responses[-1]
    assert mock_model.call_count == 3


def test_forward_without_shortcut(mock_model: Mock) -> None:
    """Test forward pass without shortcut inference."""
    adapter = MetaLadderAdapter(model=mock_model, use_shortcut=False)

    # Mock responses for each step
    responses = [
        "Problem Type: Quadratic\nSolution Method: Factoring",  # identify_problem_type
        "Similar Problem: y^2 + 3y + 2\nSolution: y = -1, -2",  # generate_meta_problem
        "Solve the quadratic equation",  # restate_problem
        "x = -2 or x = -3",  # final solution
    ]
    mock_model.side_effect = responses

    result = adapter.forward("Solve x^2 + 5x + 6 = 0")

    assert result.text == responses[-1]
    assert mock_model.call_count == 4


def test_forward_with_conversation_history(adapter: MetaLadderAdapter) -> None:
    """Test forward pass with conversation history input."""
    history: List[Dict[str, str]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Solve x^2 + 5x + 6 = 0"},
    ]

    result = adapter.forward(history)
    assert isinstance(result.text, str)

    # Should extract last user message
    adapter.model.assert_any_call(pytest.helpers.ANY(lambda x: "x^2 + 5x + 6 = 0" in x))


def test_caching_meta_problems(mock_model: Mock) -> None:
    """Test caching of meta-problems."""
    adapter = MetaLadderAdapter(model=mock_model, cache_size=10)

    # Setup mock responses
    responses = [
        "Problem Type: Quadratic\nSolution Method: Factoring",
        "Similar Problem: y^2 + 3y + 2\nSolution: y = -1, -2",
        "Restated problem",
        "Final solution",
    ]
    mock_model.side_effect = responses

    # First call should generate everything
    result1 = adapter.forward("Solve x^2 + 5x + 6 = 0")
    assert mock_model.call_count == 4

    # Reset mock for second call
    mock_model.reset_mock()
    mock_model.side_effect = responses

    # Second call should use cached values
    result2 = adapter.forward("Solve x^2 + 5x + 6 = 0")
    assert mock_model.call_count == 1  # Only final solution generation

    assert result1.text == result2.text


def test_cache_normalization(mock_model: Mock) -> None:
    """Test that similar questions use the same cache key."""
    adapter = MetaLadderAdapter(model=mock_model)

    # Different whitespace and capitalization
    questions = ["Solve x^2 + 5x + 6 = 0", "  Solve x^2 + 5x + 6 = 0  ", "SOLVE x^2 + 5x + 6 = 0"]

    # All should generate the same cache key
    keys = [adapter._get_cache_key(q) for q in questions]
    assert len(set(keys)) == 1


def test_clear_cache(mock_model: Mock) -> None:
    """Test cache clearing functionality."""
    adapter = MetaLadderAdapter(model=mock_model)

    # Setup mock responses
    responses = [
        "Problem Type: Quadratic\nSolution Method: Factoring",
        "Similar Problem: y^2 + 3y + 2\nSolution: y = -1, -2",
        "Restated problem",
        "Final solution",
    ]
    mock_model.side_effect = responses

    # First call to populate cache
    adapter.forward("Solve x^2 + 5x + 6 = 0")
    assert len(adapter._meta_problems) == 1

    # Clear cache
    adapter.clear_cache()
    assert len(adapter._meta_problems) == 0

    # Next call should regenerate everything
    mock_model.reset_mock()
    mock_model.side_effect = responses
    adapter.forward("Solve x^2 + 5x + 6 = 0")
    assert mock_model.call_count == 4


def test_optimizer_integration(mock_model: Mock, mock_optimizer: Mock) -> None:
    """Test integration with DSPy optimizers."""
    adapter = MetaLadderAdapter(model=mock_model, optimizer=mock_optimizer, use_shortcut=True)

    # Setup mock responses
    responses = ["Problem Type: Quadratic\nSolution Method: Factoring", "Restated problem", "Final solution"]
    mock_model.side_effect = responses

    result = adapter.forward("Solve x^2 + 5x + 6 = 0")

    # Verify optimizer was called
    mock_optimizer.optimize.assert_called_once()
    assert isinstance(mock_optimizer.optimize.call_args[0][0], str)

    # Verify final solution uses optimized prompt
    assert mock_model.call_args_list[-1][0][0] == "Optimized prompt"


def test_hash_meta_problem() -> None:
    """Test MetaProblem hash functionality."""
    problem1 = MetaProblem(question="q1", solution="s1", problem_type="t1", solution_method="m1")
    problem2 = MetaProblem(question="q1", solution="s1", problem_type="t1", solution_method="m1")
    problem3 = MetaProblem(question="q2", solution="s2", problem_type="t2", solution_method="m2")

    # Same content should produce same hash
    assert hash(problem1) == hash(problem2)
    # Different content should produce different hash
    assert hash(problem1) != hash(problem3)

    # Should work as dictionary keys
    cache = {}
    cache[problem1] = "result1"
    assert cache[problem2] == "result1"  # Can retrieve with equal object
