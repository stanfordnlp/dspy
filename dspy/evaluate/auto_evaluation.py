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


class DecompositionalSemanticRecallPrecision(dspy.Signature):
    """
    Compare a system's response to the ground truth to compute recall and precision of key ideas.
    You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision.
    """

    question: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    ground_truth_key_ideas: str = dspy.OutputField(desc="enumeration of key ideas in the ground truth")
    system_response_key_ideas: str = dspy.OutputField(desc="enumeration of key ideas in the system response")
    discussion: str = dspy.OutputField(desc="discussion of the overlap between ground truth and system response")
    recall: float = dspy.OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = dspy.OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


def f1_score(precision, recall):
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(dspy.Module):
    def __init__(self, threshold=0.66, decompositional=False):
        self.threshold = threshold

        if decompositional:
            self.module = dspy.ChainOfThought(DecompositionalSemanticRecallPrecision)
        else:
            self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def forward(self, example, pred, trace=None):
        scores = self.module(question=example.question, ground_truth=example.response, system_response=pred.response)
        score = f1_score(scores.precision, scores.recall)

        return score if trace is None else score >= self.threshold



###########


class AnswerCompleteness(dspy.Signature):
    """
    Estimate the completeness of a system's responses, against the ground truth.
    You will first enumerate key ideas in each response, discuss their overlap, and then report completeness.
    """

    question: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    ground_truth_key_ideas: str = dspy.OutputField(desc="enumeration of key ideas in the ground truth")
    system_response_key_ideas: str = dspy.OutputField(desc="enumeration of key ideas in the system response")
    discussion: str = dspy.OutputField(desc="discussion of the overlap between ground truth and system response")
    completeness: float = dspy.OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")



class AnswerGroundedness(dspy.Signature):
    """
    Estimate the groundedness of a system's responses, against real retrieved documents written by people.
    You will first enumerate whatever non-trivial or check-worthy claims are made in the system response, and then
    discuss the extent to which some or all of them can be deduced from the retrieved context and basic commonsense.
    """

    question: str = dspy.InputField()
    retrieved_context: str = dspy.InputField()
    system_response: str = dspy.InputField()
    system_response_claims: str = dspy.OutputField(desc="enumeration of non-trivial or check-worthy claims in the system response")
    discussion: str = dspy.OutputField(desc="discussion of how supported the claims are by the retrieved context")
    groundedness: float = dspy.OutputField(desc="fraction (out of 1.0) of system response supported by the retrieved context")


class CompleteAndGrounded(dspy.Module):
    def __init__(self, threshold=0.66):
        self.threshold = threshold
        self.completeness_module = dspy.ChainOfThought(AnswerCompleteness)
        self.groundedness_module = dspy.ChainOfThought(AnswerGroundedness)

    def forward(self, example, pred, trace=None):
        completeness = self.completeness_module(question=example.question, ground_truth=example.response, system_response=pred.response)
        groundedness = self.groundedness_module(question=example.question, retrieved_context=pred.context, system_response=pred.response)
        score = f1_score(groundedness.groundedness, completeness.completeness)

        return score if trace is None else score >= self.threshold
