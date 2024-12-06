import dspy
from optimizer_tester import OptimizerTester
from dspy.teleprompt import MetaKNNFewShot
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key_openai = os.getenv('OPENAI_API_KEY')

if not api_key_openai:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def meta_knn_optimizer_caller(default_program, trainset, devset, test_name, dataset_name, kwargs):
    """
    Caller function for MetaKNNFewShot optimizer that follows OptimizerTester's requirements.
    """
    # Initialize the teleprompter with our MetaKNNFewShot optimizer
    teleprompter = MetaKNNFewShot(
        k=20,  # Number of nearest neighbors
        trainset=trainset,
        n_programs=16,  # Number of different programs to generate
        metric=kwargs["metric"],
        max_labeled_demos=8,
        max_rounds=1
    )

    # Compile the program using our optimizer
    compiled_program = teleprompter.compile(
        default_program.deepcopy(),
        teacher=default_program.deepcopy(),
        trainset=trainset
    )

    # Add custom information to the output dictionary
    output = {
        "test_name": f"meta_knn-{dataset_name}-{test_name}",
        "meta_prompt_style": "knn",
        "fewshot_before": True,
        "fewshot_after": True,
        "fewshot_candidates_num": 5,  # k value
        "max_bootstrapped_demos": 8,  # max_labeled_demos value
        "bootstrapping": True,
        "additional_notes": "Using MetaKNNFewShot optimizer with multiple program selection"
    }

    return compiled_program, output
def main():
    try:
        # dspy setup
        import dspy
        from optimizer_tester import OptimizerTester

        lm = dspy.LM(model="openai/gpt-4o-mini", api_key=api_key_openai)
        embedder = dspy.Embedder(
            model="openai/text-embedding-3-small",
            api_key=api_key_openai
        )
        dspy.settings.configure(lm=lm, embedder=embedder)

        # Initialize the tester
        tester = OptimizerTester(
            task_model = lm, prompt_model = lm,
        )

        # Then test MetaKNNFewShot optimizer
        print("\nRunning MetaKNNFewShot optimizer tests...")
        tester.test_optimizer_default(
            meta_knn_optimizer_caller,
            datasets=["hotpotqa", "gsm8k"],
            test_name="meta_knn_v1"
        )

    except ConnectionError as e:
        print(f"Connection error occurred: {e}")
        print("Please check your language model server configuration")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main() 