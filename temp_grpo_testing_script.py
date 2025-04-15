
import os
os.environ["OPENAI_API_KEY"] = input("OPENAI_API_KEY: ")
import dspy

dspy_lm = dspy.LM(model="openai/gpt-4o", max_tokens=None, max_completion_tokens=16384, api_key=os.environ["OPENAI_API_KEY"])
dspy.configure(lm=dspy_lm)

class CustomModule(dspy.Module):
    def __init__(self):
        self.modules = [dspy.ChainOfThought(f"question{i} -> consice_answer") for i in range(3)]
    
    def forward(self, question):
        answers = []
        for idx, module in enumerate(self.modules):
            for _ in range(idx+1):
                answers.append(module(**{f"question{idx}": question}))
        # Find the most common answer
        answer_counts = {}
        for answer in answers:
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1
        most_common_answer = max(answer_counts, key=answer_counts.get)
        return most_common_answer

dataset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the capital of Germany?", answer="Berlin").with_inputs("question"),
]

def metric(example, prediction, trace=None):
    return example.answer == prediction.consice_answer

module = CustomModule()

import importlib
import dspy
importlib.reload(dspy)
import dspy.clients.arbor.arbor
importlib.reload(dspy.clients.arbor.arbor)
from dspy.teleprompt.grpo import GRPO
importlib.reload(dspy.teleprompt.grpo)
from dspy.teleprompt.grpo import GRPO

compiler = GRPO(
    metric=metric,
    multitask=True,
    num_dspy_examples_per_grpo_step=2,
    exclude_demos=True,
    num_train_steps=5
)

compiler.compile(module, dataset)
