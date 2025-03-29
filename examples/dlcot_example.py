"""
DLCoT Example: Optimizing Long Chain-of-Thought Distillation

This example demonstrates how to use the DLCoT (Deconstructing Long Chain-of-Thought)
optimizer for enhanced distillation of long CoT data. The optimizer processes CoT data
through intelligent segmentation, redundancy elimination, and error correction,
which typically reduces token usage by 70-90% while maintaining performance.

Based on research in "Deconstructing Long Chain-of-Thought: A Structured Reasoning
Optimization Framework for Long CoT Distillation" (Luo et al., 2025).
"""

import dspy
from dspy.primitives.example import Example
from dspy.teleprompt import DLCoT, BootstrapFewShot
from typing import Dict, List, Any

# Configure DSPy with an LM (use your preferred provider)
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


# Define a simple arithmetic reasoning module
class ArithmeticReasoner(dspy.Module):
    """A module for arithmetic reasoning using chain-of-thought.

    This module implements a simple reasoning approach for solving
    arithmetic problems by leveraging chain-of-thought prompting.
    """

    def __init__(self) -> None:
        """Initialize the arithmetic reasoning module."""
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        """Perform reasoning on the input question.

        Args:
            question: The arithmetic question to solve

        Returns:
            A prediction containing the answer and reasoning
        """
        return self.prog(question=question)


# Create a training dataset with long CoT examples
trainset = [
    Example(
        question="What is 25 * 32?",
        cot="""
        To calculate 25 * 32, I'll use several approaches.
        
        Approach 1: Direct multiplication
        25 * 32 = (25 * 30) + (25 * 2)
        25 * 30 = 750
        25 * 2 = 50
        750 + 50 = 800
        
        Alternatively, I could use a different approach.
        
        Approach 2: Factor-based multiplication
        25 = 5^2
        32 = 2^5
        25 * 32 = 5^2 * 2^5 = 5^2 * 2^5 = 800
        
        Let me verify this result.
        25 * 32 = 800
        I can double-check: 800 ÷ 25 = 32 ✓
        
        Therefore, 25 * 32 = 800.
        """,
        answer="800",
    ).with_inputs("question"),
    Example(
        question="What is the area of a circle with radius 4?",
        cot="""
        I need to find the area of a circle with radius 4.
        
        The formula for the area of a circle is A = πr².
        
        Step 1: Identify what I know.
        - Radius (r) = 4
        - Formula: A = πr²
        
        Step 2: Substitute the value of radius into the formula.
        A = π * 4²
        A = π * 16
        A = 16π
        
        Alternatively, I could use the numerical value of π.
        A = 16 * 3.14159...
        A = 50.27...
        
        Let me verify this calculation.
        Area = πr² = π * 4² = 16π ≈ 50.27 square units.
        
        Therefore, the area of the circle is 16π or approximately 50.27 square units.
        """,
        answer="16π",
    ).with_inputs("question"),
]

# Create evaluation set
evalset = [
    Example(
        question="What is 48 * 25?",
        answer="1200",
    ).with_inputs("question"),
    Example(
        question="What is the area of a circle with radius 6?",
        answer="36π",
    ).with_inputs("question"),
]


def metric(example: Example, pred: dspy.Prediction, trace: Any = None) -> bool:
    """Check if the predicted answer contains the correct answer.

    This metric works with both string and numeric answers, and handles
    special cases like π in mathematical expressions.

    Args:
        example: The reference example with the correct answer
        pred: The model's prediction to evaluate
        trace: Optional execution trace (not used)

    Returns:
        True if the predicted answer contains the correct answer, False otherwise
    """
    if hasattr(example, "answer") and isinstance(example.answer, str):
        correct_answer = example.answer.strip().lower()
        predicted = pred.answer.strip().lower()

        # Handle special cases like π
        if "π" in correct_answer and "pi" in predicted:
            predicted = predicted.replace("pi", "π")

        return correct_answer in predicted
    return False


def evaluate_model(model: dspy.Module, dataset: List[Example], metric_fn: callable) -> float:
    """Manually evaluate a model on a dataset using the given metric function.

    Args:
        model: The model to evaluate
        dataset: Dataset of examples to evaluate on
        metric_fn: Function to calculate metrics

    Returns:
        Accuracy of the model on the dataset (0.0-1.0)
    """
    total_score = 0
    for example in dataset:
        try:
            prediction = model(**example.inputs())
            score = metric_fn(example, prediction)
            total_score += score
        except Exception as e:
            print(f"Error evaluating example: {e}")
            continue

    accuracy = total_score / len(dataset) if len(dataset) > 0 else 0
    return accuracy


def main() -> None:
    """Run the DLCoT optimization demonstration."""
    # Create a student model to optimize
    student_model = ArithmeticReasoner()

    # Create a DLCoT optimizer
    optimizer = DLCoT(
        metric=metric,
        redundancy_threshold=0.8,
        remove_incorrectness=False,
        distillation_optimizer=BootstrapFewShot(metric=metric, max_bootstrapped_demos=2),
        num_threads=4,
    )

    # Compare baseline and optimized models
    print("=" * 70)
    print("DLCoT: Deconstructing Long Chain-of-Thought Optimization Demo")
    print("=" * 70)

    print("\nEvaluating baseline model...")
    baseline_accuracy = evaluate_model(student_model, evalset, metric)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}")

    # Get baseline tokens for a test example
    test_question = "What is 18 * 12?"
    baseline_prediction = student_model(question=test_question)
    baseline_length = len(baseline_prediction.answer)
    print(f"\nBaseline reasoning length: {baseline_length} characters")

    # Compile the student model
    print("\nCompiling the student model with DLCoT optimization...")
    optimized_model = optimizer.compile(student_model, trainset)

    # Evaluate the optimized model
    print("\nEvaluating the optimized model...")
    optimized_accuracy = evaluate_model(optimized_model, evalset, metric)
    print(f"Optimized Accuracy: {optimized_accuracy:.2f}")

    # Try a specific example
    print("\nRunning inference on a test example...")
    optimized_prediction = optimized_model(question=test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {optimized_prediction.answer}")
    optimized_length = len(optimized_prediction.answer)
    print(f"Optimized reasoning length: {optimized_length} characters")

    # Calculate token reduction
    reduction = (1 - optimized_length / baseline_length) * 100
    print(f"Token reduction: {reduction:.1f}%")

    print("\nSummary of DLCoT Benefits:")
    print("1. More efficient CoT responses with less redundancy")
    print("2. Maintained or improved performance accuracy")
    print("3. Enhanced reasoning through structured optimization")
    print("=" * 70)


if __name__ == "__main__":
    main()
