import dspy


class SemanticRecallPrecision(dspy.Signature):
    """
    Compare a system's response to the ground truth to compute its recall and precision.
    If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.
    """

    question: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    recall: float = dspy.OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = dspy.OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


def f1_score(precision, recall):
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(dspy.Module):
    def __init__(self, threshold=0.66):
        self.threshold = threshold
        self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def forward(self, example, pred, trace=None):
        scores = self.module(question=example.question, ground_truth=example.response, system_response=pred.response)
        score = f1_score(scores.precision, scores.recall)

        return score if trace is None else score >= self.threshold


"""
Soon-to-be deprecated Signatures & Modules Below.
"""


class AnswerCorrectnessSignature(dspy.Signature):
    """Verify that the predicted answer matches the gold answer."""

    question = dspy.InputField()
    gold_answer = dspy.InputField(desc="correct answer for question")
    predicted_answer = dspy.InputField(desc="predicted answer for question")
    is_correct = dspy.OutputField(desc="True or False")


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
    is_faithful = dspy.OutputField(desc="True or False")


class AnswerFaithfulness(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate_faithfulness = dspy.ChainOfThought(AnswerFaithfulnessSignature)

    def forward(self, context, question, answer):
        return self.evaluate_faithfulness(context=context, question=question, answer=answer)
