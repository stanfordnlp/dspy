import re

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


class ComputedSemanticRecallPrecision(Signature):
    """
    Compare a system's response to the ground truth by enumerating key ideas and identifying which are covered.
    List each key idea on its own numbered line (1. idea, 2. idea, ...).
    For covered ideas, list ONLY the numbers of ideas that are present in the other response, e.g. "1, 3, 4".
    If no ideas are covered, write "None".
    """

    question: str = InputField()
    ground_truth: str = InputField()
    system_response: str = InputField()
    ground_truth_key_ideas: str = OutputField(desc="numbered list (1. 2. 3. ...) of key ideas in the ground truth")
    system_response_key_ideas: str = OutputField(
        desc="numbered list (1. 2. 3. ...) of key ideas in the system response"
    )
    ground_truth_ideas_covered_by_system: str = OutputField(
        desc="numbers of ground truth ideas covered by the system response, e.g. 1, 3, 4"
    )
    system_response_ideas_grounded_in_truth: str = OutputField(
        desc="numbers of system response ideas supported by the ground truth, e.g. 1, 2"
    )


def _count_numbered_items(text: str) -> int:
    """Count items in a numbered list like '1. foo\\n2. bar'."""
    return len(re.findall(r"^\s*\d+[.\)]\s", text, re.MULTILINE))


def _count_mentioned_numbers(text: str) -> int:
    """Count distinct numbers mentioned in text like '1, 3, 4'. Returns 0 for 'None' or empty."""
    if not text or text.strip().lower() == "none":
        return 0
    return len(set(re.findall(r"\d+", text)))


def f1_score(precision, recall):
    """Compute the F1 score from precision and recall, clamping both to [0, 1]."""
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(Module):
    """Computes semantic F1 between a prediction and ground truth via LLM-based precision/recall.

    Args:
        threshold: Minimum F1 score to accept during optimization. Defaults to 0.66.
        decompositional: Controls how precision and recall are obtained.

            - ``False`` (default): the LLM directly outputs precision and recall floats.
            - ``True``: the LLM enumerates key ideas and discusses overlap, then outputs
              precision and recall floats.
            - ``"computed"``: the LLM enumerates key ideas and identifies which are covered;
              precision and recall are computed programmatically from the counts, avoiding
              LLM float-calibration issues.
    """

    def __init__(self, threshold=0.66, decompositional=False):
        self.threshold = threshold
        self._computed = decompositional == "computed"

        if self._computed:
            self.module = ChainOfThought(ComputedSemanticRecallPrecision)
        elif decompositional:
            self.module = ChainOfThought(DecompositionalSemanticRecallPrecision)
        else:
            self.module = ChainOfThought(SemanticRecallPrecision)

    def forward(self, example, pred, trace=None):
        scores = self.module(question=example.question, ground_truth=example.response, system_response=pred.response)

        if self._computed:
            gt_count = _count_numbered_items(scores.ground_truth_key_ideas)
            sr_count = _count_numbered_items(scores.system_response_key_ideas)
            gt_covered = _count_mentioned_numbers(scores.ground_truth_ideas_covered_by_system)
            sr_grounded = _count_mentioned_numbers(scores.system_response_ideas_grounded_in_truth)

            recall = gt_covered / gt_count if gt_count > 0 else 0.0
            precision = sr_grounded / sr_count if sr_count > 0 else 0.0
            score = f1_score(precision, recall)
        else:
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
