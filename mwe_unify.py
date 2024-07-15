import os

from dotenv import load_dotenv

import dsp
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

load_dotenv()
unify_api_key = os.getenv("UNIFY_KEY")

endpoint = dsp.Unify(
    endpoint="gpt-3.5-turbo@openai",
    max_tokens=150,
    api_key=unify_api_key,
)

dspy.settings.configure(lm=endpoint)

# Load math questions from the GSM8K dataset.
gsm8k = GSM8K()

print("Loading GSM8K train and dev sets")

gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

print(f"train set: {gsm8k_trainset}")


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question) -> str:
        return self.prog(question=question)


# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = {"max_bootstrapped_demos": 4, "max_labeled_demos": 4}

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

endpoint.inspect_history(n=1)

print(
    """Done! This example showcases how to set up your environment, define a custom module,
    compile a model, and rigorously evaluate its performance using the provided dataset and teleprompter configurations.
    """,
)
