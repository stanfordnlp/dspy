from dspy.dsp.utils import EM, normalize_text

from dspy.evaluate.metrics import answer_exact_match, answer_passage_match
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.auto_evaluation import SemanticF1, CompleteAndGrounded
