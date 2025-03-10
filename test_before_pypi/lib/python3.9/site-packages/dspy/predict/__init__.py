from dspy.predict.aggregation import majority
from dspy.predict.best_of_n import BestOfN
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.chain_of_thought_with_hint import ChainOfThoughtWithHint
from dspy.predict.knn import KNN
from dspy.predict.multi_chain_comparison import MultiChainComparison
from dspy.predict.predict import Predict
from dspy.predict.program_of_thought import ProgramOfThought
from dspy.predict.react import ReAct, Tool
from dspy.predict.refine import Refine
from dspy.predict.parallel import Parallel

__all__ = [
    "majority",
    "BestOfN",
    "ChainOfThought",
    "ChainOfThoughtWithHint",
    "KNN",
    "MultiChainComparison",
    "Predict",
    "ProgramOfThought",
    "ReAct",
    "Refine",
    "Tool",
    "Parallel",
]
