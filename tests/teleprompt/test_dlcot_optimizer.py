"""
Tests for the DLCoT (Deconstructing Long Chain-of-Thought) optimizer.
"""

from typing import List, Dict, Any, TYPE_CHECKING

import dspy
import pytest
from dspy.primitives.example import Example
from dspy.teleprompt import DLCoT, BootstrapFewShot

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class SimpleModule(dspy.Module):
    """A simple module for testing purposes."""
    
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str) -> dspy.Prediction:
        return self.prog(question=question)


@pytest.fixture
def sample_examples() -> List[Example]:
    """Return a list of sample examples for testing."""
    return [
        Example(
            question="What is 25 * 32?",
            cot="""
            To calculate 25 * 32, I'll use several approaches.
            
            Approach 1: Direct multiplication
            25 * 32 = (25 * 30) + (25 * 2)
            25 * 30 = 750
            25 * 2 = 50
            750 + 50 = 800
            
            Approach 2: Factor-based multiplication
            25 = 5^2
            32 = 2^5
            25 * 32 = 5^2 * 2^5 = 800
            
            Therefore, 25 * 32 = 800.
            """,
            answer="800"
        ).with_inputs("question"),
        Example(
            question="What is the area of a circle with radius 4?",
            cot="""
            I need to find the area of a circle with radius 4.
            
            The formula for the area of a circle is A = πr².
            
            A = π * 4²
            A = π * 16
            A = 16π
            
            Therefore, the area of the circle is 16π.
            """,
            answer="16π"
        ).with_inputs("question"),
    ]


def simple_metric(example: Example, pred: dspy.Prediction, trace: Any = None) -> bool:
    """A simple metric for testing."""
    correct_answer = example.answer.strip().lower()
    predicted = pred.answer.strip().lower()
    return correct_answer in predicted


def test_dlcot_initialization() -> None:
    """Test that the DLCoT optimizer can be initialized with default parameters."""
    optimizer = DLCoT()
    assert optimizer.redundancy_threshold == 0.8
    assert optimizer.remove_incorrectness is False
    assert isinstance(optimizer.segment_config, dict)
    assert optimizer.distillation_optimizer is None
    
    # Test with custom parameters
    custom_optimizer = DLCoT(
        metric=simple_metric,
        redundancy_threshold=0.7,
        remove_incorrectness=True,
        segment_config={"max_segments": 3},
        distillation_optimizer=BootstrapFewShot(metric=simple_metric),
        num_threads=2
    )
    assert custom_optimizer.metric == simple_metric
    assert custom_optimizer.redundancy_threshold == 0.7
    assert custom_optimizer.remove_incorrectness is True
    assert custom_optimizer.segment_config["max_segments"] == 3
    assert isinstance(custom_optimizer.distillation_optimizer, BootstrapFewShot)
    assert custom_optimizer.num_threads == 2


def test_find_cot_field(sample_examples: List[Example]) -> None:
    """Test the _find_cot_field method."""
    optimizer = DLCoT()
    
    # Test with an example that has a 'cot' field
    assert optimizer._find_cot_field(sample_examples[0]) == "cot"
    
    # Test with an example that has a different field with multi-line content
    example_with_reasoning = Example(
        question="What is 2+2?",
        reasoning="""
        I'll add 2 and 2. 
        Step 1: Add the numbers.
        2 + 2 = 4.
        Therefore, the result is 4.
        """,
        answer="4"
    ).with_inputs("question")
    assert optimizer._find_cot_field(example_with_reasoning) == "reasoning"
    
    # Test with an example that has no CoT fields
    example_without_cot = Example(
        question="What is 2+2?",
        answer="4"
    ).with_inputs("question")
    assert optimizer._find_cot_field(example_without_cot) is None


def test_segment_cot() -> None:
    """Test the _segment_cot method."""
    optimizer = DLCoT()
    
    # Test with a simple CoT content
    cot_content = """
    Let me solve this problem step by step.
    
    Approach 1: Direct calculation
    2 + 2 = 4
    
    Therefore, the answer is 4.
    """
    
    segments = optimizer._segment_cot(cot_content)
    assert "Problem_Understand" in segments
    assert "Approach_Explore" in segments
    assert "Conclusion" in segments
    assert "Direct calculation" in segments["Approach_Explore"]
    assert "the answer is 4" in segments["Conclusion"]


def test_analyze_redundancy() -> None:
    """Test the redundancy analysis logic."""
    optimizer = DLCoT(redundancy_threshold=0.5)
    
    # Create test approaches with redundancy
    approaches = [
        {"id": "approach_1", "content": "Using algebra", "type": "algebraic"},
        {"id": "approach_2", "content": "Using geometry", "type": "geometric"},
        {"id": "approach_3", "content": "Using more algebra", "type": "algebraic"},
    ]
    
    analyzed = optimizer._analyze_redundancy_and_correctness(approaches)
    
    # Check that approach_3 is marked as redundant (it's the second algebraic approach)
    assert not analyzed[0]["is_redundant"]
    assert not analyzed[1]["is_redundant"]
    assert analyzed[2]["is_redundant"]
    
    # All approaches should be marked as correct since they don't contain error markers
    assert analyzed[0]["is_correct"]
    assert analyzed[1]["is_correct"]
    assert analyzed[2]["is_correct"]


def test_optimize_integration() -> None:
    """Test the optimization integration logic."""
    # Create a DLCoT optimizer with redundancy removal forced by high threshold
    optimizer = DLCoT(redundancy_threshold=0.1, remove_incorrectness=True)
    
    # Create test segments
    segments = {
        "Question_Repeat": "What is 2+2?",
        "Problem_Understand": "I need to calculate the sum.",
        "Approach_Explore": "Approach 1: Addition\n2+2=4\n\nApproach 2: Counting\n1,2,3,4",
        "Verify": "Let me verify: 2+2=4 ✓",
        "Conclusion": "Therefore, 2+2=4."
    }
    
    # Create analyzed approaches with redundancy and errors
    analyzed_approaches = [
        {"id": "approach_1", "content": "Approach 1: Addition\n2+2=4", "type": "arithmetic", 
         "is_redundant": False, "is_correct": True, "redundancy_score": 0.0},
        {"id": "approach_2", "content": "Approach 2: Counting\n1,2,3,4", "type": "arithmetic", 
         "is_redundant": True, "is_correct": False, "redundancy_score": 0.9},
    ]
    
    optimized = optimizer._optimize_integration(segments, analyzed_approaches)
    
    # In this case, the second approach should be removed because it's both redundant and incorrect
    assert optimized["Approach_Explore"] == "Approach 1: Addition\n2+2=4"
    
    # Other segments should remain unchanged
    assert optimized["Question_Repeat"] == segments["Question_Repeat"]
    assert optimized["Conclusion"] == segments["Conclusion"]


def test_ensure_coherence() -> None:
    """Test the coherence reconstruction logic."""
    optimizer = DLCoT()
    
    # Create test segments
    segments = {
        "Question_Repeat": "What is 2+2?",
        "Problem_Understand": "I need to calculate the sum.",
        "Approach_Explore": "2+2=4",
        "Verify": "Let me verify: 2+2=4 ✓",
        "Conclusion": "Therefore, 2+2=4."
    }
    
    coherent_content = optimizer._ensure_coherence(segments)
    
    # All segments should be present in the coherent content
    assert "What is 2+2?" in coherent_content
    assert "calculate the sum" in coherent_content
    assert "2+2=4" in coherent_content
    assert "verify" in coherent_content
    assert "Therefore" in coherent_content
    
    # Segments should be properly separated
    assert coherent_content.count("\n\n") >= 4


def test_process_example(sample_examples: List[Example]) -> None:
    """Test the example processing logic."""
    optimizer = DLCoT(redundancy_threshold=0.5)
    
    # Process a sample example
    processed = optimizer._process_example(sample_examples[0])
    
    # Ensure the example is processed but still contains key information
    assert hasattr(processed, "cot")
    assert "25 * 32" in processed.cot
    assert "800" in processed.cot
    
    # Test processing an example without CoT
    example_without_cot = Example(
        question="What is 2+2?",
        answer="4"
    ).with_inputs("question")
    
    # Should return the original example unchanged
    processed_no_cot = optimizer._process_example(example_without_cot)
    assert processed_no_cot.question == "What is 2+2?"
    assert processed_no_cot.answer == "4" 