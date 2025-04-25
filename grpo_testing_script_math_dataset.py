# os.environ["OPENAI_API_KEY"] = input("OPENAI_API_KEY: ")
import dspy
import multiprocessing
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

# Load MATH dataset
from dspy.datasets import MATH
dataset = MATH(subset='algebra')
train_dataset = dataset.train[:20]
test_dataset = dataset.dev[:20]
print(len(dataset.train), len(dataset.dev))

# Test base model on one sample
example = train_dataset[0]
print("Question:", example.question)
print("Answer:", example.answer)

class MATHSignature(dspy.Signature):
    question = dspy.InputField(format=str)
    answer = dspy.OutputField(format=str)

module = dspy.ChainOfThought(MATHSignature)
module.set_lm(student_lm)
o = module(question=example.question)
print("Base model predicted answer:", o.answer)

import dspy
from dspy.teleprompt.grpo import GRPO

train_kwargs = {
    "temperature": 0.9,
    "beta": 0.04,
    "update_interval": 1,
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
    metric=dataset.metric,
    multitask=True,
    num_dspy_examples_per_grpo_step=5,
    num_samples_per_input=4,  # should we test this with 1?
    exclude_demos=True,
    num_train_steps=100,
    use_train_as_val=False,
    train_kwargs=train_kwargs,
    num_threads=multiprocessing.cpu_count(),
    num_steps_for_val=2
)

compiler.compile(
    student=module,
    trainset=train_dataset,
    valset=test_dataset
)
