import re
import dspy
from dspy import Example
from dspy.teleprompt import LabeledFewShot, BootstrapFewShotWithRandomSearch

import datasets

# pipeline configs
turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo)


# load up the gsm8k dataset from hub into a dspy-compatible format
dataset = datasets.load_dataset("gsm8k", 'main')
training_data = dataset["train"].shuffle(seed=42).select(range(200))
dev_data = dataset["train"].shuffle(seed=42).select(range(200, 500))

trainset = [Example(x).with_inputs("question") for x in training_data]
devset = [Example(x).with_inputs("question") for x in dev_data]
testset = [Example(x).with_inputs("question") for x in dataset["test"]]

# validation function
def gsm8k_accuracy(example, pred):
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
    print("GT:", gt_answer, "PRED:", pred.answer, "PRED-ANS:", pred_answer)
    
    return pred_answer == gt_answer

class ThoughtReflection(dspy.Module):
    def __init__(self, num_attempts):
        self.predict = dspy.ChainOfThought("question -> answer", n=num_attempts)
        self.compare = dspy.MultiChainComparison("question -> answer", M=num_attempts)
        
    def forward(self, question):
        completions = self.predict(question=question).completions
        answer = self.compare(question=question, completions=completions)
        return answer


# teleprompter = LabeledFewShot(k=8)
teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_accuracy)    

prog = ThoughtReflection(num_attempts=5)
compiled_prog = teleprompter.compile(prog, trainset=trainset, valset=devset)

my_question = testset[0].question
pred = compiled_prog(my_question)

print("Question:", my_question)
turbo.inspect_history(n=1)

# TODO: Debug the parsing of the output!