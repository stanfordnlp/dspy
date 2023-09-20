import os

import dspy
from dsp.utils import deduplicate
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot

# pipeline configs
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)


# load dataset
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]


# signatures of dspy modules
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

# validation logic for verifying traces
def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred):
        return False
    
    if not dspy.evaluate.answer_passage_match(example, pred):
        return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100:
        return False
    
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): # 4
        return False

    return True


########################## NEW STUFF ##########################


# declaration of dspy program
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query

            # assertion
            dspy.Assert(lambda x: len(x) <= 1, query)

            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)

        return dspy.Prediction(context=context, answer=pred.answer)

# compile dspy program using a teleprompter (optimizer)
teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)

student = SimplifiedBaleen()
teacher = SimplifiedBaleen(passages_per_hop=2)
compiled_baleen = teleprompter.compile(student, teacher=teacher, trainset=trainset)

my_question = "How many storeys are in the castle that David Gregory inherited?"
pred = compiled_baleen(my_question)
turbo.inspect_history(n=3)