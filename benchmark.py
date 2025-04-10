"""Benchmark comparing ChainOfThought with MetaLadder."""
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import dspy
from dspy.primitives import Module
from dspy.adapters import MetaLadderAdapter
from dspy.clients.lm import LM

# Set up the language model with API key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Configure language model
lm = LM(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Disable caching
dspy.settings.configure(cache_seed=None)

class MathSolver(dspy.Signature):
    """Signature for solving math problems."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="numerical answer with units")
    reasoning = dspy.OutputField(desc="step by step reasoning")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.
    
    Attributes:
        accuracy: Percentage of correct solutions
        avg_time: Average time per problem in seconds
        problem_types: Dictionary mapping problem types to their accuracies
        generalization_score: Score for similar but slightly modified problems
    """
    accuracy: float
    avg_time: float
    problem_types: Dict[str, float]
    generalization_score: float


def get_test_problems() -> Dict[str, List[Tuple[str, str]]]:
    """Get test problems with expected answers.
    
    Returns:
        Dictionary mapping problem types to lists of (problem, answer) tuples
    """
    return {
        "multiplication": [
            (
                "If a train travels at 60 miles per hour for 2.5 hours, how far does it travel?",
                "150 miles"
            ),
            (
                "A factory produces 120 widgets per hour. How many widgets does it produce in 8 hours?",
                "960 widgets"
            )
        ],
        "division": [
            (
                "If 144 cookies are divided equally among 3 charity events, how many cookies does each event get?",
                "48 cookies"
            ),
            (
                "A company has $900 to divide among 6 employees. How much does each employee receive?",
                "$150"
            )
        ]
    }


def get_variation_problems() -> Dict[str, List[Tuple[str, str]]]:
    """Get variation problems to test generalization.
    
    Returns:
        Dictionary mapping problem types to lists of (problem, answer) tuples
    """
    return {
        "multiplication": [
            (
                "A cyclist pedals at 15 kilometers per hour for 3.5 hours. What distance does the cyclist cover?",
                "52.5 kilometers"
            )
        ],
        "division": [
            (
                "If 288 candies need to be distributed equally to 4 schools, how many candies does each school get?",
                "72 candies"
            )
        ]
    }


def run_benchmark(
    model: Module,
    problems: List[Tuple[str, str]],
    model_name: str
) -> Tuple[int, float]:
    """Run benchmark on a set of problems.
    
    Args:
        model: The model to benchmark
        problems: List of (problem, expected_answer) tuples
        model_name: Name of the model for logging
        
    Returns:
        Tuple of (correct_count, total_time)
    """
    correct = 0
    total_time = 0
    
    for i, (problem, expected) in enumerate(problems, 1):
        print(f"\nProblem {i}:")
        print(f"Question: {problem}")
        print(f"Expected: {expected}")
        
        start_time = time.time()
        result = model(question=problem)
        answer = result.answer
        time_taken = time.time() - start_time
        
        print(f"{model_name} answer: {answer}")
        if hasattr(result, "reasoning"):
            print(f"Reasoning: {result.reasoning}")
            
        if expected.lower() in answer.lower():
            correct += 1
            print("✓ Correct")
        else:
            print("✗ Incorrect")
            
        total_time += time_taken
        print(f"Time: {time_taken:.2f}s")
    
    return correct, total_time


def benchmark_models() -> None:
    """Run benchmark comparing ChainOfThought and MetaLadder."""
    # Create solvers
    cot_solver = dspy.ChainOfThought(MathSolver)
    meta_solver = MetaLadderAdapter(cot_solver)
    
    # Get test problems
    problems = get_test_problems()
    total_problems = sum(len(probs) for probs in problems.values())
    
    print("=== Model Comparison Benchmark ===\n")
    
    # Test Chain of Thought
    print("Chain of Thought:")
    for prob_type, test_cases in problems.items():
        correct, time_taken = run_benchmark(cot_solver, test_cases, "Chain of Thought")
        print(f"\n{prob_type.title()}:")
        print(f"Accuracy: {(correct / len(test_cases)) * 100:.1f}%")
        print(f"Average time: {time_taken / len(test_cases):.2f}s")
    
    # Test MetaLadder
    print("\nMetaLadder:")
    for prob_type, test_cases in problems.items():
        correct, time_taken = run_benchmark(meta_solver, test_cases, "MetaLadder")
        print(f"\n{prob_type.title()}:")
        print(f"Accuracy: {(correct / len(test_cases)) * 100:.1f}%")
        print(f"Average time: {time_taken / len(test_cases):.2f}s")


if __name__ == "__main__":
    benchmark_models() 