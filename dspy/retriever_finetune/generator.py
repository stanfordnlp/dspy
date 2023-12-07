import dspy
from dsp.utils import deduplicate
from dspy.retriever_finetune import GenerateAnswer, TEMPERATURE


class RetrievalFinetuneGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer, temperature=TEMPERATURE)
    
    def forward(self, question, context):
        context = deduplicate(context)
        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)