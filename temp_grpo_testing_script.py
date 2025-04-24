# os.environ["OPENAI_API_KEY"] = input("OPENAI_API_KEY: ")
import dspy
from dspy.clients.lm_local_arbor import ArborProvider

# Assume that the server is running at the following port
port = 1111
arbor_api_base = f"http://localhost:{port}/v1/"
api_key = "arbor"
provider = ArborProvider()
# Uncomment this line if you want to use LM.finetune(...)
# This is a hack for the time being, we will modify the "finetune" method to
# take in an LM instead.
# dspy.settings.arbor_base_api = arbor_api_base  
# provider = LocalProvider()

student_lm_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
student_lm = dspy.LM(model=f"openai/arbor:{student_lm_name}", provider=provider, temperature=0.7, api_base=arbor_api_base, api_key=api_key)
student_lm.launch()

print(student_lm("Hi there! How are you?"))

class CustomSignature(dspy.Signature):
    """
    Say a random joke and then answer the question.
    """
    question = dspy.InputField(format=str)
    joke = dspy.OutputField()
    concise_answer = dspy.OutputField(format=str)

class CustomModule(dspy.Module):
    def __init__(self):
        self.modules = [dspy.ChainOfThought(CustomSignature) for i in range(3)]
    
    def forward(self, question):
        answers = []
        for idx, module in enumerate(self.modules):
            for _ in range(idx+1):
                answers.append(module(**{"question": (" " * (idx+1)) + question}))
        
        max_joke_length = max([len(answer.joke) for answer in answers])
        concise_answers = [answer.concise_answer for answer in answers]
        most_common_answer = max(set(concise_answers), key=concise_answers.count)
        return dspy.Prediction(
            concise_answer=most_common_answer,
            max_joke_length=max_joke_length,
        )

dataset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the capital of Germany?", answer="Berlin").with_inputs("question"),
] * 6

def metric(example, prediction, trace=None):
    score1 = 1 if example.answer == prediction.concise_answer else 0
    score2 = 1 / prediction.max_joke_length if prediction.max_joke_length > 0 else -10000
    return (score1 + score2) / 2

module = CustomModule()
module.set_lm(student_lm)

import importlib
import dspy
importlib.reload(dspy)
# import dspy.clients.arbor.arbor
# importlib.reload(dspy.clients.arbor.arbor)
from dspy.teleprompt.grpo import GRPO
importlib.reload(dspy.teleprompt.grpo)

train_kwargs = {
    "temperature": 0.9,
    "beta": 0.04,
    "update_interval": 25,
    # Note that we don't add the "num_generations" here. dspy.GRPO computes this
    # as follows. We don't currentl support teacher programs and instead use the
    # the student program as the teacher program, so len(teachers) is always 1
    # for now.
    #
    #     num_generations = num_samples_per_input * len(teachers)
    #
}

dspy.settings.experimental = True
compiler = GRPO(
    metric=metric,
    multitask=True,
    num_dspy_examples_per_grpo_step=2,
    num_samples_per_input=4,  # should we test this with 1?
    exclude_demos=True,
    num_train_steps=2,
    use_train_as_val=True,
    train_kwargs=train_kwargs
)

compiler.compile(
    student=module,
    trainset=dataset,
)
