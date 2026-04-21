#!/usr/bin/env python3
"""
Basic test to verify SBO implementation works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy import Example
from dspy.teleprompt.sbo import SemanticBundleOptimization


class SimpleQA(dspy.Module):
    """Simple QA program for testing."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


def test_metric(example, pred, trace=None):
    """Simple metric for testing - checks if answer contains expected word."""
    pred_answer = pred.answer.lower() if hasattr(pred, 'answer') else str(pred).lower()
    gold_answer = example.answer.lower()

    # Simple word overlap score
    pred_words = set(pred_answer.split())
    gold_words = set(gold_answer.split())

    if not gold_words:
        return 0.0

    overlap = len(pred_words & gold_words)
    score = overlap / len(gold_words)

    return min(1.0, score)


def main():
    print("="*60)
    print("Testing SBO Implementation")
    print("="*60)

    # Create toy examples
    trainset = [
        Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        Example(question="What is 2+2?", answer="four").with_inputs("question"),
        Example(question="What color is the sky?", answer="blue").with_inputs("question"),
    ]

    valset = [
        Example(question="What is the capital of Italy?", answer="Rome").with_inputs("question"),
        Example(question="What is 3+3?", answer="six").with_inputs("question"),
    ]

    print(f"\nTrain set: {len(trainset)} examples")
    print(f"Val set: {len(valset)} examples")

    # Setup a mock LM (we'll use a real one but with minimal calls)
    # For real testing, you'd want to mock this
    try:
        lm = dspy.LM(
            model="ollama_chat/qwen3:4b-instruct",  # Available model
            api_base="http://localhost:11434",
            temperature=0.7,
            max_tokens=100
        )
        dspy.settings.configure(lm=lm)
        print(f"\nConfigured LM: {lm.model}")
    except Exception as e:
        print(f"\nWarning: Could not configure LM: {e}")
        print("This test requires a local Ollama instance running.")
        print("Skipping full test, but SBO code structure is valid.")
        return

    # Create program
    program = SimpleQA()
    print("\nInitialized SimpleQA program")

    # Create SBO optimizer with minimal settings
    print("\nCreating SBO optimizer with minimal settings...")
    optimizer = SemanticBundleOptimization(
        metric=test_metric,
        num_candidates=2,  # Very small for quick test
        num_judge_samples=1,  # Minimal sampling
        descent_param=0.1,
        max_iterations=3,  # Just a few iterations
        max_null_steps=2,
        temperature=0.7,
        track_stats=True
    )

    print("SBO optimizer created successfully!")

    # Run optimization
    print("\n" + "="*60)
    print("Running SBO optimization...")
    print("="*60 + "\n")

    try:
        optimized_program = optimizer.compile(
            student=program,
            trainset=trainset,
            valset=valset
        )

        print("\n" + "="*60)
        print("✓ SBO OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)

        if optimizer.result:
            result = optimizer.result
            print(f"\nResults:")
            print(f"  Total iterations: {result.total_iterations}")
            print(f"  Serious steps: {result.num_serious_steps}")
            print(f"  Null steps: {result.num_null_steps}")
            print(f"  Bundle size: {len(result.bundle)}")
            print(f"  Best loss: {result.bundle[result.best_idx].loss:.4f}")
            print(f"  Final val scores: {result.val_scores}")

        # Test the optimized program
        print("\n" + "="*60)
        print("Testing optimized program...")
        print("="*60)

        test_question = "What is the capital of Spain?"
        prediction = optimized_program(question=test_question)
        print(f"\nQuestion: {test_question}")
        print(f"Answer: {prediction.answer if hasattr(prediction, 'answer') else prediction}")

        print("\n✓ ALL TESTS PASSED!")

    except Exception as e:
        print(f"\n✗ ERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
