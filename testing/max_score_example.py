import os
from dotenv import load_dotenv
import dspy
from max_score_tester import BootstrapMaxScoreTester
from optimizer_tester import OptimizerTester
from phoenix.otel import register
from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key_openai = os.getenv('OPENAI_API_KEY')

if not api_key_openai:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Setup Phoenix tracing
register(
    endpoint="https://app.phoenix.arize.com/v1/traces",
)

# Initialize OpenTelemetry instrumentation
DSPyInstrumentor().instrument(skip_dep_check=True)
LiteLLMInstrumentor().instrument(skip_dep_check=True)

lm = dspy.LM(model="openai/gpt-4o-mini", api_key=api_key_openai)
embedder = dspy.Embedder(
    model="openai/text-embedding-3-small",
    api_key=api_key_openai
)
dspy.settings.configure(lm=lm, embedder=embedder)

# Initialize the tester
tester = BootstrapMaxScoreTester(
    n_programs=2,
    max_labeled_demos=1,
    early_stopping_threshold=0.95,
    num_threads=32,
    dataset_name="hover_retrieve_discrete"
)

# Load dataset
optimizer_tester = OptimizerTester(task_model=lm, prompt_model=lm)
task = optimizer_tester.load_dataset("hover_retrieve_discrete")

# Run the test
results = tester.test_dataset(task)

print(f"Maximum Train Score: {results['train_results']['solved_items']:.3f}")
print(f"Maximum Dev Score: {results['dev_results']['solved_items']:.3f}")
print(f"Maximum Test Score: {results['test_results']['solved_items']:.3f}")

