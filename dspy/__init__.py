import dspy.retrievers
from dspy.dsp.utils.settings import settings
from dspy.utils.asyncify import asyncify
from dspy.utils.logging_utils import configure_dspy_loggers, disable_logging, enable_logging
from dspy.utils.saving import load
from dspy.utils.streaming import streamify

from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *

# Functional must be imported after primitives, predict and signatures
from .functional import *  # isort: skip
from dspy.evaluate import Evaluate  # isort: skip
from dspy.clients import *  # isort: skip
from dspy.adapters import *  # isort: skip


configure_dspy_loggers(__name__)

from dspy.dsp.colbertv2 import ColBERTv2

# from dspy.dsp.you import You

configure = settings.configure
context = settings.context


import dspy.teleprompt

LabeledFewShot = dspy.teleprompt.LabeledFewShot
BootstrapFewShot = dspy.teleprompt.BootstrapFewShot
BootstrapFewShotWithRandomSearch = dspy.teleprompt.BootstrapFewShotWithRandomSearch
BootstrapRS = dspy.teleprompt.BootstrapFewShotWithRandomSearch
BootstrapFinetune = dspy.teleprompt.BootstrapFinetune
BetterTogether = dspy.teleprompt.BetterTogether
COPRO = dspy.teleprompt.COPRO
MIPROv2 = dspy.teleprompt.MIPROv2
Ensemble = dspy.teleprompt.Ensemble
