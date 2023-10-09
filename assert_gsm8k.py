import re
import dspy
from dspy import Example
from dspy.teleprompt import LabeledFewShot, BootstrapFewShotWithRandomSearch

import datasets

from tool import calculator, prefix_calculator

# pipeline configs
turbo = dspy.OpenAI(model="gpt-3.5-turbo-16k")
dspy.settings.configure(lm=turbo)


# load up the gsm8k dataset from hub into a dspy-compatible format
dataset = datasets.load_dataset("gsm8k", 'main')
training_data = dataset["train"].shuffle(seed=42).select(range(10))
dev_data = dataset["train"].shuffle(seed=42).select(range(10, 15))

trainset = [Example(x).with_inputs("question") for x in training_data]
devset = [Example(x).with_inputs("question") for x in dev_data]
testset = [Example(x).with_inputs("question") for x in dataset["test"]]

# validation function
def gsm8k_accuracy(example, pred, trace=None):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"
    
    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    gt_answer = extract_answer(example.answer)
    assert gt_answer != INVALID_ANS
    
    pred_answer = extract_answer(pred.answer)
    
    return pred_answer == gt_answer


class GenerateAnswer(dspy.Signature):
    """Answer questions with a single numeric value."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="#### ${answer}")
    

########## PROGRAMS ##########

class Vanilla(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GenerateAnswer)
    
    def forward(self, question):
        return self.predict(question=question)


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        return self.predict(question=question)


class ThoughtReflection(dspy.Module):
    def __init__(self, num_attempts):
        super().__init__()

        self.predict = dspy.ChainOfThought(GenerateAnswer, n=num_attempts)
        self.compare = dspy.MultiChainComparison(GenerateAnswer, M=num_attempts)
        
    def forward(self, question):
        completions = self.predict(question=question).completions        
        answer = self.compare(question=question, completions=completions)
        return answer

########## COMPILATION (w/ random hyperparameter search) ##########

prog = Vanilla()
# prog = CoT()
# prog = ThoughtReflection(num_attempts=5)

teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_accuracy)
compiled_prog = teleprompter.compile(prog, trainset=trainset, valset=devset)

########## EVALUATION ##########

my_question = testset[0].question
pred = compiled_prog(my_question)

print("Question:", my_question)
turbo.inspect_history(n=1)
