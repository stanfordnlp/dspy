from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *

import dspy.retrievers

# Functional must be imported after primitives, predict and signatures
from .functional import *  # isort: skip
from dspy.evaluate import Evaluate  # isort: skip
from dspy.clients import *  # isort: skip
from dspy.adapters import *  # isort: skip
from dspy.utils.logging_utils import configure_dspy_loggers, disable_logging, enable_logging
from dspy.utils.asyncify import asyncify
from dspy.utils.saving import load

from dspy.dsp.utils.settings import settings

configure_dspy_loggers(__name__)

from dspy.dsp.modules.colbertv2 import ColBERTv2
# from dspy.dsp.modules.you import You

configure = settings.configure
context = settings.context

from dspy.utils.cache import Cache
from dspy.clients import DISK_CACHE_DIR, DISK_CACHE_LIMIT
MEM_CACHE_LIMIT = float(os.environ.get("DSPY_CACHE_LIMIT", float("inf")))  # unlimited by default

# Initialize the cache
dspy_cache = Cache(
    directory=os.path.join(DISK_CACHE_DIR, ".cache_v2_6"),
    disk_size_limit=DISK_CACHE_LIMIT,
    mem_size_limit=MEM_CACHE_LIMIT
)
settings.configure(cache=dspy_cache)

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
