
import os
# os.environ["OPENAI_API_KEY"] = input("OPENAI_API_KEY: ")
import dspy
from dspy.clients.lm_local_arbor import ArborProvider
from dspy.clients.lm_local import LocalProvider

base_url = "http://localhost:8000/v1/"
provider = ArborProvider(api_base=base_url)
# provider = LocalProvider()

# student_lm_name = "meta-llama/Llama-3.2-1B-Instruct"
student_lm_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
student_lm = dspy.LM(model=f"openai/arbor:{student_lm_name}", provider=provider, api_key="EMPTY", temperature=0.7)
student_lm.launch({"api_base": base_url})
dspy.configure(lm=student_lm)

# dspy_lm = dspy.LM(model="openai/gpt-4o", max_tokens=None, max_completion_tokens=16384, api_key=os.environ["OPENAI_API_KEY"])

print(student_lm("Hi there! How are you?"))

class CustomSignature(dspy.Signature):
    """
    Say a random joke and then answer the question.
    """
    question = dspy.InputField(format=str)
    joke = dspy.OutputField()
    consice_answer = dspy.OutputField(format=str)

class CustomModule(dspy.Module):
    def __init__(self):
        self.modules = [dspy.ChainOfThought(CustomSignature) for i in range(3)]
    
    def forward(self, question):
        answers = []
        for idx, module in enumerate(self.modules):
            for _ in range(idx+1):
                answers.append(module(**{f"question": (" " * (idx+1)) + question}))
        
        max_joke_length = max([len(answer.joke) for answer in answers])
        concise_answers = [answer.consice_answer for answer in answers]
        most_common_answer = max(set(concise_answers), key=concise_answers.count)
        return dspy.Prediction(
            consice_answer=most_common_answer,
            max_joke_length=max_joke_length,
        )

dataset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the capital of Germany?", answer="Berlin").with_inputs("question"),
] * 6

def metric(example, prediction, trace=None):
    score1 = 1 if example.answer == prediction.consice_answer else 0
    score2 = 1 / prediction.max_joke_length if prediction.max_joke_length > 0 else -10000
    return (score1 + score2) / 2

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
    num_train_steps=100,
    use_train_as_val=True,
    num_rollouts_per_dspy_example_per_step=8
)

compiler.compile(module, dataset)
