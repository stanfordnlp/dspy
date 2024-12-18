from typing import Literal

from datasets import load_dataset

import dspy
from dspy.clients.databricks import DatabricksProvider

# Define the range as a tuple of valid integers
CLASSES = tuple(range(77))

ds = load_dataset("PolyAI/banking77")
trainset_hf = ds["train"][:100]
trainset = []

for text, label in zip(trainset_hf["text"], trainset_hf["label"]):
    # Each example should have two fields, `inputs` and `answer`, with `inputs` as the input field,
    # and `answer` as the output field.
    trainset.append(dspy.Example(text=text, answer=label).with_inputs("text"))

gold = {text: label for text, label in zip(trainset_hf["text"], trainset_hf["label"])}

lm = dspy.LM(
    model="databricks/databricks-meta-llama-3-1-70b-instruct",
    provider=DatabricksProvider,
    finetuning_model="meta-llama/Llama-3.2-3B",
)

dspy.settings.configure(lm=lm)
dspy.settings.experimental = True


def accuracy(example, pred, trace=None):
    return int(example.answer == int(pred.answer))


class Classify(dspy.Signature):
    """As a part of a banking issue traiging system, classify the intent of a natural language query."""

    text = dspy.InputField()
    answer: Literal[CLASSES] = dspy.OutputField()


class Program(dspy.Module):
    def __init__(self, oracle=False):
        self.oracle = oracle
        self.classify = dspy.ChainOfThoughtWithHint(Classify)

    def forward(self, text):
        if self.oracle and text in gold:
            hint = f"the right label is {gold[text]}"
        else:
            hint = None
        return self.classify(text=text, hint=hint)


model = Program(oracle=True)
print("Try the original model: ", model("I am still waiting on my card?"))

train_kwargs = {
    "train_data_path": "/Volumes/main/chenmoney/testing/dspy_testing/classification",
    "register_to": "main.chenmoney.finetuned_model_classification",
    "task_type": "CHAT_COMPLETION",
}

optimized = dspy.BootstrapFinetune(metric=accuracy, num_threads=10, train_kwargs=train_kwargs).compile(
    student=model, trainset=trainset
)
optimized.oracle = False

print("Try the optimized model: ", optimized("I am still waiting on my card?"))
