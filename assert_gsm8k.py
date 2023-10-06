import dspy
from dsp.utils import deduplicate
from dspy.teleprompt import LabeledFewShot, BootstrapFewShotWithRandomSearch

import datasets

# pipeline configs
turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo)


# load up the gsm8k dataset from the huggingface hub
dataset = datasets.load_dataset("gsm8k")

# split into train and test
trainset = dataset["train"]
testset = dataset["test"]

print(trainset)

# # each example should have a question and answer
# trainset = []
# testset = []

# def gsm8k_accuracy(example, pred, trace=None):
#     if not dspy.evaluate.answer_exact_match(example, pred):
#         return False
#     return True

# class ThoughtReflection(dspy.Module):
#     def __init__(self, num_attempts):
#         self.predict = dspy.ChainOfThought("question -> answer", n=num_attempts)
#         self.compare = dspy.MultiChainComparison("question -> answer", M=num_attempts)
        
#     def forward(self, question):
#         completions = self.predict(question=question).completions
#         answer = self.compare(question=question, completions=completions)
#         return answer


# telemprompter = LabeledFewShot(k=8)
# # teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_accuracy)    


# prog = ThoughtReflection(num_attempts=5)
# compiled_prog = telemprompter.compile(prog, trainset=trainset)