from dspy.teleprompt.avatar_optimizer import AvatarOptimizer
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data
from dspy.teleprompt.copro_optimizer import COPRO
from dspy.teleprompt.ensemble import Ensemble
from dspy.teleprompt.infer_rules import InferRules
from dspy.teleprompt.knn_fewshot import KNNFewShot
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.simba import SIMBA
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.teleprompt_optuna import BootstrapFewShotWithOptuna
from dspy.teleprompt.vanilla import LabeledFewShot

try:
    from .gepa.gepa import GEPA
except ModuleNotFoundError as error:
    if error.name != "gepa":
        raise

    class GEPA:
        """Stand-in for GEPA when the gepa package is not installed.

        DSPy installs gepa only on Python < 3.15, because gepa declares requires-python <3.15.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gepa is required to use dspy.GEPA. It is not installed by default on Python 3.15 and later "
                "because gepa declares requires-python <3.15."
            )

__all__ = [
    "AvatarOptimizer",
    "BetterTogether",
    "BootstrapFewShot",
    "BootstrapFinetune",
    "COPRO",
    "Ensemble",
    "GEPA",
    "KNNFewShot",
    "MIPROv2",
    "BootstrapFewShotWithRandomSearch",
    "BootstrapFewShotWithOptuna",
    "LabeledFewShot",
    "InferRules",
    "SIMBA",
    "bootstrap_trace_data",
]
