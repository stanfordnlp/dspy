from max_score_tester import BootstrapMaxScoreTester, SimpleMaxScoreTester
from optimizer_tester import OptimizerTester

import dspy
from optimizer_tester import OptimizerTester
from dspy.teleprompt import MetaKNNFewShot
from dotenv import load_dotenv
import os

from tests.dsp_LM.teleprompt.test_bootstrap import trainset

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key_openai = os.getenv('OPENAI_API_KEY')

if not api_key_openai:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

lm = dspy.LM(model="openai/gpt-4o-mini", api_key=api_key_openai)
embedder = dspy.Embedder(
    model="openai/text-embedding-3-small",
    api_key=api_key_openai
)
dspy.settings.configure(lm=lm, embedder=embedder)

# Initialize the tester
tester = BootstrapMaxScoreTester(
    n_programs=64,
    max_labeled_demos=16,
    early_stopping_threshold=0.95,
)

# Load dataset
optimizer_tester = OptimizerTester(task_model=lm, prompt_model=lm)
task = optimizer_tester.load_dataset("hotpotqa")

# Run the test
results = tester.test_dataset(task)

print(f"Maximum Train Score: {results['train_results']['solved_items']:.3f}")
print(f"Maximum Dev Score: {results['dev_results']['solved_items']:.3f}")
print(f"Maximum Test Score: {results['test_results']['solved_items']:.3f}")

