from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures import InputField, OutputField, Signature


class SemanticRecallPrecision(Signature):
    """
    Compare a system's response to the ground truth to compute its recall and precision.
    If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.
    """

    question: str = InputField()
    ground_truth: str = InputField()
    system_response: str = InputField()
    recall: float = OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


class DecompositionalSemanticRecallPrecision(Signature):
    """
    Compare a system's response to the ground truth to compute recall and precision of key ideas.
    You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision.
    """

    question: str = InputField()
    ground_truth: str = InputField()
    system_response: str = InputField()
    ground_truth_key_ideas: str = OutputField(desc="enumeration of key ideas in the ground truth")
    system_response_key_ideas: str = OutputField(desc="enumeration of key ideas in the system response")
    discussion: str = OutputField(desc="discussion of the overlap between ground truth and system response")
    recall: float = OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


def f1_score(precision, recall):
    """Compute the F1 score from precision and recall, clamping both to [0, 1]."""
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(Module):
    """Computes semantic F1 between a prediction and ground truth via LLM-based precision/recall.

    Args:
        threshold: Minimum F1 score to accept during optimization. Defaults to 0.66.
        decompositional: If True, uses DecompositionalSemanticRecallPrecision. Defaults to False.
    """

    def __init__(self, threshold=0.66, decompositional=False):
        self.threshold = threshold

        if decompositional:
            self.module = ChainOfThought(DecompositionalSemanticRecallPrecision)
        else:
            self.module = ChainOfThought(SemanticRecallPrecision)

    def forward(self, example, pred, trace=None):
        scores = self.module(question=example.question, ground_truth=example.response, system_response=pred.response)
        score = f1_score(scores.precision, scores.recall)

        return Prediction(score=score if trace is None else score >= self.threshold)


###########


class AnswerCompleteness(Signature):
    """
    Estimate the completeness of a system's responses, against the ground truth.
    You will first enumerate key ideas in each response, discuss their overlap, and then report completeness.
    """

    question: str = InputField()
    ground_truth: str = InputField()
    system_response: str = InputField()
    ground_truth_key_ideas: str = OutputField(desc="enumeration of key ideas in the ground truth")
    system_response_key_ideas: str = OutputField(desc="enumeration of key ideas in the system response")
    discussion: str = OutputField(desc="discussion of the overlap between ground truth and system response")
    completeness: float = OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")


class AnswerGroundedness(Signature):
    """
    Estimate the groundedness of a system's responses, against real retrieved documents written by people.
    You will first enumerate whatever non-trivial or check-worthy claims are made in the system response, and then
    discuss the extent to which some or all of them can be deduced from the retrieved context and basic commonsense.
    """

    question: str = InputField()
    retrieved_context: str = InputField()
    system_response: str = InputField()
    system_response_claims: str = OutputField(
        desc="enumeration of non-trivial or check-worthy claims in the system response"
    )
    discussion: str = OutputField(desc="discussion of how supported the claims are by the retrieved context")
    groundedness: float = OutputField(
        desc="fraction (out of 1.0) of system response supported by the retrieved context"
    )


class CompleteAndGrounded(Module):
    """Combines answer completeness and groundedness into a single score.

    Args:
        threshold: Minimum score to accept during optimization. Defaults to 0.66.
    """

    def __init__(self, threshold=0.66):
        self.threshold = threshold
        self.completeness_module = ChainOfThought(AnswerCompleteness)
        self.groundedness_module = ChainOfThought(AnswerGroundedness)

    def forward(self, example, pred, trace=None):
        completeness = self.completeness_module(
            question=example.question, ground_truth=example.response, system_response=pred.response
        )
        groundedness = self.groundedness_module(
            question=example.question, retrieved_context=pred.context, system_response=pred.response
        )
        score = f1_score(groundedness.groundedness, completeness.completeness)

        return Prediction(score=score if trace is None else score >= self.threshold)


###########


class RAGGroundedRefusal(Module):
    """Scores a RAG response for correctness, groundedness, and refusal behavior, with textual feedback.

    Returns ``Prediction(score, feedback)``, where ``feedback`` names the failure mode so that
    feedback-driven optimizers (e.g. ``dspy.GEPA``) can reflect on it. Whether an example is
    answerable from its context is read from the structured ``example.answerable`` field; whether
    the prediction refused is read from ``pred.refused`` if present, otherwise via ``is_refusal``.

    Args:
        threshold: Minimum score to accept during optimization. Defaults to 0.66.
        is_refusal: Optional callable applied to ``pred.response`` to detect refusal when
            ``pred.refused`` is not set.
    """

    def __init__(self, threshold=0.66, is_refusal=None):
        self.threshold = threshold
        self.is_refusal = is_refusal
        self.correctness_module = ChainOfThought(SemanticRecallPrecision)
        self.groundedness_module = ChainOfThought(AnswerGroundedness)

    def forward(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        refused = getattr(pred, "refused", None)
        if refused is None:
            if self.is_refusal is None:
                raise ValueError(
                    "RAGGroundedRefusal needs a refusal signal: set `pred.refused` (bool) or "
                    "pass `is_refusal=` to detect it from `pred.response`."
                )
            refused = bool(self.is_refusal(pred.response))

        if not example.answerable:
            if refused:
                score, feedback = 1.0, "The response correctly refused: the context does not support an answer."
            else:
                score, feedback = (
                    0.0,
                    "The response answered although the context does not support an answer; it should refuse.",
                )
        elif refused:
            score, feedback = 0.0, "The response refused although the context supports an answer."
        else:
            correctness = self.correctness_module(
                question=example.question, ground_truth=example.response, system_response=pred.response
            )
            groundedness = self.groundedness_module(
                question=example.question, retrieved_context=pred.context, system_response=pred.response
            )
            correctness_score = f1_score(correctness.precision, correctness.recall)
            groundedness_score = max(0.0, min(1.0, groundedness.groundedness))
            score = f1_score(correctness_score, groundedness_score)

            issues = []
            if correctness_score < self.threshold:
                issues.append(f"misses or contradicts the ground truth (expected: {example.response})")
            if groundedness_score < self.threshold:
                issues.append("makes claims not supported by the retrieved context")
            if issues:
                feedback = f"The response {' and '.join(issues)}."
            else:
                feedback = "The response is correct and grounded in the retrieved context."

        return Prediction(score=score if trace is None else score >= self.threshold, feedback=feedback)
