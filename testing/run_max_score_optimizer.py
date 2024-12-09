from testing.optimizer_tester import OptimizerTester
from testing.optimizers.max_score_optimizer import max_score_optimizer_caller
import sys
import dspy

def main():
    try:
        # Create OpenAI client configuration
        task_model = dspy.OpenAI(
            model="gpt-4o-mini",
            max_tokens=150
        )

        # Initialize the tester with OpenAI configuration
        tester = OptimizerTester(
            default_train_num=None,   # Use full training set
            default_dev_num=10,       # Small dev set
            default_test_num=10,      # Only 10 test samples
            num_threads=16,           # Increased thread count for parallel processing
            max_errors=10,            # Stop after 10 errors to fail fast
            prompt_model_name="gpt-4o-mini",  # Using gpt-4o-mini
            task_model_name="gpt-4o-mini",    # Same model for tasks
            task_model=task_model,            # Pass the configured OpenAI client
            prompt_model=task_model           # Use same model for prompts
        )

        # Run max score optimizer with parallel processing
        print("\nRunning max score optimizer...")
        tester.test_optimizer_default(
            max_score_optimizer_caller,
            datasets=["hotpotqa_conditional"],
            test_name="max_score_full_train"
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()