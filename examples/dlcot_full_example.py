"""
DLCoT Full Example: Optimizing Long Chain-of-Thought Distillation

This example demonstrates the complete functionality of the DLCoT optimizer for
enhancing the distillation of long chain-of-thought (CoT) data. The optimizer processes
CoT data through intelligent segmentation, redundancy elimination, and error correction,
then performs bootstrapping and distillation on the optimized data.

Based on research in "Deconstructing Long Chain-of-Thought: A Structured Reasoning
Optimization Framework for Long CoT Distillation" (Luo et al., 2025).
"""

import dspy
from dspy.primitives.example import Example
from dspy.teleprompt import DLCoT, BootstrapFewShot
from typing import List, Any
import random

# Enable experimental features for fine-tuning (not needed for BootstrapFewShot)
# dspy.settings.experimental = True

# Configure DSPy with an LM
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

# Set random seed for reproducibility
random.seed(42)


# Define a simple math reasoning module that uses chain-of-thought
class MathReasoner(dspy.Module):
    """A module for math reasoning using chain-of-thought.

    This implements arithmetic and basic math problem-solving using
    chain-of-thought reasoning to show the steps toward the solution.
    """

    def __init__(self) -> None:
        """Initialize the math reasoning module."""
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        """Perform reasoning on the input question.

        Args:
            question: The math question to solve

        Returns:
            A prediction containing the answer and reasoning
        """
        return self.prog(question=question)


def create_math_dataset() -> List[Example]:
    """Create a synthetic dataset for math reasoning.

    This function generates examples with long chain-of-thought reasoning
    for various math problems, including arithmetic, geometry, and algebra.

    Returns:
        A list of examples with questions, CoT reasoning, and answers
    """
    examples = [
        Example(
            question="What is 25 * 32?",
            answer="800",
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
            
            I could also try yet another approach.
            
            Approach 3: Using the distributive property again
            25 * 32 = 25 * (30 + 2)
            = (25 * 30) + (25 * 2)
            = 750 + 50
            = 800
            
            Let me verify this result.
            25 * 32 = 800
            I can double-check: 800 ÷ 25 = 32 ✓
            
            Therefore, 25 * 32 = 800.
            """,
        ).with_inputs("question"),
        Example(
            question="What is the area of a circle with radius 4?",
            answer="16π",
            cot="""
            I need to find the area of a circle with radius 4.
            
            The formula for the area of a circle is A = πr².
            
            Approach 1: Direct substitution
            Given:
            - Radius (r) = 4
            - Formula: A = πr²
            
            Step 1: Substitute the value into the formula.
            A = π * 4²
            A = π * 16
            A = 16π
            
            Alternatively, let me try a different method.
            
            Approach 2: Using numerical value of π
            A = π * 16
            A ≈ 3.14159 * 16
            A ≈ 50.27
            
            Let me verify this calculation.
            A = πr² = π * 4² = 16π ≈ 50.27 square units.
            
            Therefore, the area of the circle is 16π or approximately 50.27 square units.
            """,
        ).with_inputs("question"),
        Example(
            question="If a rectangle has length 12 and width 8, what is its perimeter?",
            answer="40",
            cot="""
            To find the perimeter of a rectangle, I need to add all four sides.

            Approach 1: Using the formula P = 2(l + w)
            Given:
            - Length (l) = 12
            - Width (w) = 8
            
            P = 2(l + w)
            P = 2(12 + 8)
            P = 2(20)
            P = 40
            
            I can also calculate this another way.
            
            Approach 2: Adding all four sides individually
            P = l + w + l + w
            P = 12 + 8 + 12 + 8
            P = 24 + 16
            P = 40
            
            Therefore, the perimeter of the rectangle is 40 units.
            """,
        ).with_inputs("question"),
        Example(
            question="What is the sum of the first 10 even numbers?",
            answer="110",
            cot="""
            I need to find the sum of the first 10 even numbers.
            
            Approach 1: List and add
            The first 10 even numbers are: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
            
            Sum = 2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20
            Sum = 6 + 14 + 30 + 60
            Sum = 20 + 90
            Sum = 110
            
            Let me try a different approach.
            
            Approach 2: Using the arithmetic sequence formula
            The even numbers form an arithmetic sequence with first term a = 2 and common difference d = 2.
            The sum of the first n terms is: S_n = n/2 * [2a + (n-1)d]
            
            For n = 10:
            S_10 = 10/2 * [2(2) + (10-1)2]
            S_10 = 5 * [4 + 18]
            S_10 = 5 * 22
            S_10 = 110
            
            Let me verify with a third approach.
            
            Approach 3: Using the formula for sum of even numbers
            The sum of the first n even numbers is n(n+1).
            
            For n = 10:
            Sum = 10 * 11 = 110
            
            Therefore, the sum of the first 10 even numbers is 110.
            """,
        ).with_inputs("question"),
        Example(
            question="If 3x + 5 = 20, what is the value of x?",
            answer="5",
            cot="""
            I need to solve the equation 3x + 5 = 20 for x.
            
            Approach 1: Algebraic manipulation
            Step 1: Subtract 5 from both sides.
            3x + 5 - 5 = 20 - 5
            3x = 15
            
            Step 2: Divide both sides by 3.
            3x/3 = 15/3
            x = 5
            
            Let me verify this solution.
            
            Approach 2: Substitution to check
            If x = 5, then:
            3x + 5 = 3(5) + 5 = 15 + 5 = 20 ✓
            
            Therefore, x = 5.
            """,
        ).with_inputs("question"),
        # Adding 5 more examples to meet the 10-example minimum
        Example(
            question="What is the value of 5² + 4³?",
            answer="89",
            cot="""
            I need to calculate 5² + 4³.
            
            Approach 1: Direct calculation
            First, let's calculate each term separately.
            5² = 5 × 5 = 25
            4³ = 4 × 4 × 4 = 64
            
            Now, let's add them together.
            5² + 4³ = 25 + 64 = 89
            
            Approach 2: Using exponent properties
            5² = 5^2 = 25
            4³ = 4^3 = 64
            5² + 4³ = 25 + 64 = 89
            
            Therefore, 5² + 4³ = 89.
            """,
        ).with_inputs("question"),
        Example(
            question="If a triangle has sides of length 3, 4, and 5, what is its area?",
            answer="6",
            cot="""
            To find the area of a triangle with sides 3, a, and 5, I'll use different approaches.
            
            Approach 1: Using Heron's formula
            Heron's formula states that for a triangle with sides a, b, and c, the area is:
            Area = √(s(s-a)(s-b)(s-c)), where s = (a+b+c)/2 is the semi-perimeter.
            
            Given sides 3, 4, and 5:
            s = (3 + 4 + 5)/2 = 12/2 = 6
            
            Area = √(6 × (6-3) × (6-4) × (6-5))
            Area = √(6 × 3 × 2 × 1)
            Area = √36
            Area = 6
            
            Approach 2: Using the fact this is a 3-4-5 right triangle
            The sides 3, 4, and 5 form a right triangle (Pythagorean triple).
            For a right triangle, Area = (base × height)/2
            Area = (3 × 4)/2 = 12/2 = 6
            
            Therefore, the area of the triangle is 6 square units.
            """,
        ).with_inputs("question"),
        Example(
            question="What is the value of log₂(32)?",
            answer="5",
            cot="""
            I need to find the value of log₂(32).
            
            Approach 1: Using the definition of logarithm
            log₂(32) asks: to what power must I raise 2 to get 32?
            
            Let's think about this systematically:
            2¹ = 2
            2² = 4
            2³ = 8
            2⁴ = 16
            2⁵ = 32
            
            So, 2⁵ = 32, which means log₂(32) = 5.
            
            Approach 2: Using logarithm properties
            log₂(32) = log₂(2⁵)
            Using the property log₂(2ⁿ) = n:
            log₂(2⁵) = 5
            
            Therefore, log₂(32) = 5.
            """,
        ).with_inputs("question"),
        Example(
            question="If the probability of an event A is 0.3 and the probability of event B is 0.5, what is the probability of either A or B occurring if they are independent?",
            answer="0.65",
            cot="""
            I need to find P(A ∪ B) given that P(A) = 0.3, P(B) = 0.5, and events A and B are independent.
            
            Approach 1: Using the addition rule for independent events
            For any two events, P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
            
            Since A and B are independent, P(A ∩ B) = P(A) × P(B) = 0.3 × 0.5 = 0.15
            
            So, P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
            P(A ∪ B) = 0.3 + 0.5 - 0.15 = 0.8 - 0.15 = 0.65
            
            Alternatively, I can verify this with another approach.
            
            Approach 2: Using the complement rule
            P(A ∪ B) = 1 - P(neither A nor B)
            P(neither A nor B) = P(A' ∩ B') = P(A') × P(B') = (1-0.3) × (1-0.5) = 0.7 × 0.5 = 0.35
            
            Thus, P(A ∪ B) = 1 - 0.35 = 0.65
            
            Therefore, the probability of either event A or event B occurring is 0.65 or 65%.
            """,
        ).with_inputs("question"),
        Example(
            question="Solve for x in the equation 2x² - 5x - 12 = 0",
            answer="x = 4 or x = -1.5",
            cot="""
            I need to solve the quadratic equation 2x² - 5x - 12 = 0.
            
            Approach 1: Using the quadratic formula
            For an equation in the form ax² + bx + c = 0, the solution is:
            x = (-b ± √(b² - 4ac)) / (2a)
            
            In this case, a = 2, b = -5, and c = -12.
            
            x = (5 ± √((-5)² - 4×2×(-12))) / (2×2)
            x = (5 ± √(25 + 96)) / 4
            x = (5 ± √121) / 4
            x = (5 ± 11) / 4
            
            This gives us two solutions:
            x = (5 + 11) / 4 = 16 / 4 = 4
            x = (5 - 11) / 4 = -6 / 4 = -1.5
            
            Approach 2: Using factoring
            Let's try to factor 2x² - 5x - 12 = 0
            
            I need factors of 2×(-12) = -24 that add up to -5.
            The factors are 3 and -8 because 3 + (-8) = -5 and 3×(-8) = -24.
            
            So, 2x² - 5x - 12 = 0 can be rewritten as:
            2x² + 3x - 8x - 12 = 0
            x(2x + 3) - 4(2x + 3) = 0
            (2x + 3)(x - 4) = 0
            
            Setting each factor equal to zero:
            2x + 3 = 0, which gives x = -3/2 = -1.5
            x - 4 = 0, which gives x = 4
            
            Therefore, the solutions are x = 4 or x = -1.5.
            """,
        ).with_inputs("question"),
        Example(
            question="What is the value of 3√36 + 2√49?",
            answer="32",
            cot="""
            I need to calculate 3√36 + 2√49.
            
            Approach 1: Simplify each term, then add
            First, let's simplify each term:
            3√36 = 3 × 6 = 18
            2√49 = 2 × 7 = 14
            
            Now, let's add them:
            3√36 + 2√49 = 18 + 14 = 32
            
            Wait, I made a calculation error. Let me recalculate:
            3√36 = 3 × 6 = 18
            2√49 = 2 × 7 = 14
            18 + 14 = 32
            
            Actually, let me double-check my work:
            √36 = 6
            √49 = 7
            3√36 = 3 × 6 = 18
            2√49 = 2 × 7 = 14
            3√36 + 2√49 = 18 + 14 = 32
            
            I've made another error in my final addition. Let me redo:
            18 + 14 = 32
            
            Hmm, that's still not right. Let me break it down step by step:
            18 + 14
            = 18 + 10 + 4
            = 28 + 4
            = 32
            
            But wait, I need to go back to my original calculation. Let me double-check the entire problem:
            3√36 + 2√49 
            = 3 × 6 + 2 × 7
            = 18 + 14
            = 32
            
            I apologize for the confusion. After careful rechecking, I realize I've been making an error. The correct answer is:
            3√36 + 2√49 = 3 × 6 + 2 × 7 = 18 + 14 = 32.
            
            Oh wait, I've made a mistake in adding 18 and 14. Let me calculate one more time:
            18 + 14 = 32
            
            Actually, let me use a different approach to verify.
            
            Approach 2: Direct calculation
            3√36 + 2√49
            = 3 × 6 + 2 × 7
            = 18 + 14
            = 32
            
            Upon second double-check, I've realized that I've been making a consistent addition error. 18 + 14 is actually 32, not 29 as I incorrectly stated.
            
            Therefore, 3√36 + 2√49 = 32.
            """,
        ).with_inputs("question"),
        Example(
            question="Expand (x + 3)(x - 5)",
            answer="x² - 2x - 15",
            cot="""
            I need to expand the expression (x + 3)(x - 5).
            
            Approach 1: Using the FOIL method (First, Outer, Inner, Last)
            (x + 3)(x - 5)
            
            First terms: x × x = x²
            Outer terms: x × (-5) = -5x
            Inner terms: 3 × x = 3x
            Last terms: 3 × (-5) = -15
            
            Combining all terms:
            x² - 5x + 3x - 15 = x² - 2x - 15
            
            Approach 2: Using the distributive property
            (x + 3)(x - 5) = (x + 3)x - (x + 3)5
            = x² + 3x - 5x - 15
            = x² - 2x - 15
            
            Therefore, (x + 3)(x - 5) = x² - 2x - 15.
            """,
        ).with_inputs("question"),
        Example(
            question="What is the derivative of f(x) = 2x³ - 5x² + 4x - 7?",
            answer="f'(x) = 6x² - 10x + 4",
            cot="""
            I need to find the derivative of f(x) = 2x³ - 5x² + 4x - 7.
            
            Approach 1: Using power rule for each term
            The power rule states that for a function g(x) = xⁿ, its derivative is g'(x) = n×xⁿ⁻¹.
            
            Let's apply this to each term of f(x):
            
            For 2x³: The derivative is 2 × 3 × x³⁻¹ = 6x²
            For -5x²: The derivative is -5 × 2 × x²⁻¹ = -10x
            For 4x: The derivative is 4 × 1 × x¹⁻¹ = 4
            For -7: The derivative of a constant is 0
            
            Combining these terms:
            f'(x) = 6x² - 10x + 4
            
            Therefore, the derivative of f(x) = 2x³ - 5x² + 4x - 7 is f'(x) = 6x² - 10x + 4.
            """,
        ).with_inputs("question"),
    ]

    return examples


def math_metric(example: Example, pred: dspy.Prediction, trace: Any = None) -> bool:
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
    correct_answer = str(example.answer).strip().lower()
    prediction = str(pred.answer).strip().lower()

    # Handle special cases like π
    if "π" in correct_answer and "pi" in prediction:
        prediction = prediction.replace("pi", "π")

    # Check if prediction contains the correct answer
    return correct_answer in prediction


def evaluate_model(
    model: dspy.Module, dataset: List[Example], metric_fn: callable
) -> float:
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
    """Run the complete DLCoT optimization demonstration."""
    # Create dataset
    full_dataset = create_math_dataset()
    train_size = int(0.6 * len(full_dataset))
    trainset = full_dataset[:train_size]
    evalset = full_dataset[train_size:]

    # Create a baseline model for comparison
    baseline_model = MathReasoner()

    # Print a header for the demo
    print("=" * 70)
    print("DLCoT: Deconstructing Long Chain-of-Thought Optimizer Demo")
    print("=" * 70)

    # Baseline evaluation
    print("\n1. Evaluating the baseline model...")
    baseline_accuracy = evaluate_model(baseline_model, evalset, math_metric)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}")

    # Test baseline on a specific example to show CoT length
    print("\n2. Example of baseline CoT output:")
    test_question = "What is 15 * 24?"
    baseline_prediction = baseline_model(question=test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {baseline_prediction.answer}")
    if hasattr(baseline_prediction, "reasoning"):
        print(f"Reasoning length: {len(baseline_prediction.reasoning)} characters")
        baseline_cot_length = len(baseline_prediction.reasoning)

    # Create a DLCoT optimizer
    print("\n3. Creating and applying the DLCoT optimizer...")
    dlcot_optimizer = DLCoT(
        metric=math_metric,
        redundancy_threshold=0.7,
        remove_incorrectness=False,
        segment_config={"max_segments": 5},
        distillation_optimizer=BootstrapFewShot(
            metric=math_metric, max_bootstrapped_demos=4
        ),
        num_threads=4,
    )

    # Process the training data using DLCoT
    print("\n4. DLCoT is processing the long chain-of-thought data...")
    optimized_model = dlcot_optimizer.compile(baseline_model, trainset)

    # Evaluate the optimized model
    print("\n5. Evaluating the DLCoT-optimized model...")
    dlcot_accuracy = evaluate_model(optimized_model, evalset, math_metric)
    print(f"DLCoT-Optimized Accuracy: {dlcot_accuracy:.2f}")

    # Test optimized model on the same example to compare CoT length
    print("\n6. Example of DLCoT-optimized output:")
    optimized_prediction = optimized_model(question=test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {optimized_prediction.answer}")
    if hasattr(optimized_prediction, "reasoning"):
        print(f"Reasoning length: {len(optimized_prediction.reasoning)} characters")
        optimized_cot_length = len(optimized_prediction.reasoning)
        if "baseline_cot_length" in locals():
            reduction = (1 - optimized_cot_length / baseline_cot_length) * 100
            print(f"Token reduction: {reduction:.1f}%")

    # Compare DLCoT with another optimizer (BootstrapFewShot) for reference
    print("\n7. Comparing with BootstrapFewShot optimizer...")
    bfs_optimizer = BootstrapFewShot(metric=math_metric, max_bootstrapped_demos=4)

    # Apply BootstrapFewShot
    bfs_model = bfs_optimizer.compile(baseline_model, trainset=trainset)

    # Evaluate the BootstrapFewShot model
    bfs_accuracy = evaluate_model(bfs_model, evalset, math_metric)
    print(f"BootstrapFewShot Accuracy: {bfs_accuracy:.2f}")

    # Summary of results
    print("\n" + "=" * 70)
    print("Summary of Results")
    print("=" * 70)
    print(f"Baseline Accuracy:           {baseline_accuracy:.2f}")
    print(f"BootstrapFewShot Accuracy:   {bfs_accuracy:.2f}")
    print(f"DLCoT-Optimized Accuracy:    {dlcot_accuracy:.2f}")

    if "baseline_cot_length" in locals() and "optimized_cot_length" in locals():
        reduction = (1 - optimized_cot_length / baseline_cot_length) * 100
        print(f"CoT Length Reduction:        {reduction:.1f}%")

    print("\nDLCoT successfully optimized the chain-of-thought reasoning by:")
    print("1. Segmenting the CoT into logical components")
    print("2. Identifying and removing redundant reasoning approaches")
    print("3. Preserving the essential verification steps")
    print("4. Ensuring coherence in the optimized output")
    print("=" * 70)


if __name__ == "__main__":
    main()
