import dspy


class AnswerCorrectnessSignature(dspy.Signature):
    """Verify that the predicted answer matches the gold answer."""

    question = dspy.InputField()
    gold_answer = dspy.InputField(desc="correct answer for question")
    predicted_answer = dspy.InputField(desc="predicted answer for question")
    is_correct = dspy.OutputField(desc='True or False')

class AnswerCorrectness(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate_correctness = dspy.ChainOfThought(AnswerCorrectnessSignature)
    
    def forward(self, question, gold_answer, predicted_answer):
        return self.evaluate_correctness(question=question, gold_answer=gold_answer, predicted_answer=predicted_answer)


class AnswerFaithfulnessSignature(dspy.Signature):
    """Verify that the predicted answer is based on the provided context."""
    
    context = dspy.InputField(desc="relevant facts for producing answer")
    question = dspy.InputField()
    answer = dspy.InputField(desc="often between 1 and 5 words")
    is_faithful = dspy.OutputField(desc='True or False')

class AnswerFaithfulness(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate_faithfulness = dspy.ChainOfThought(AnswerFaithfulnessSignature)
    
    def forward(self, context, question, answer):
        return self.evaluate_faithfulness(context=context, question=question, answer=answer)
