import dspy 

class AnswerCorrectness(dspy.Module):
    class Signature(dspy.Signature):
        """Checks if predicted answer for question is correct"""
        question = dspy.InputField()
        answer = dspy.InputField(desc="predicted answer for question")
        correct = dspy.OutputField(desc='True or False')
    
    def __init__(self):
        super().__init__()
        self.evaluate_correctness = dspy.ChainOfThought(self.Signature())
    
    def forward(self, question, answer):
        return self.evaluate_correctness(question=question, answer=answer)

class AnswerFaithfulness(dspy.Module):
    class Signature(dspy.Signature):
        """Checks if predicted answer for question is based on rationale"""
        context = dspy.InputField(desc="relevant facts for producing predicted answer")
        question = dspy.InputField()
        answer = dspy.InputField(desc="predicted answer for question")
        faithful = dspy.OutputField(desc='True or False')

    def __init__(self):
        super().__init__()
        self.evaluate_faithfulness = dspy.ChainOfThought(self.Signature())

    def forward(self, context, question, answer):
        return self.evaluate_faithfulness(context=context, question=question, answer=answer)
