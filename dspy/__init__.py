import dsp
from dsp.modules.hf_client import (
    ChatModuleClient,
    HFClientVLLM,
    HFServerTGI,
)

from dsp.modules.schemas import *

from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *
from .utils.logging import logger, set_log_output

# Functional must be imported after primitives, predict and signatures
from .functional import *  # isort: skip

settings = dsp.settings

LM = dsp.LM

LLM = dsp.LLM
Encoder = dsp.Encoder

ColBERTv2 = dsp.ColBERTv2
ColBERTv2RerankerLocal = dsp.ColBERTv2RerankerLocal
ColBERTv2RetrieverLocal = dsp.ColBERTv2RetrieverLocal
Pyserini = dsp.PyseriniRetriever

HFClientTGI = dsp.HFClientTGI
HFClientVLLM = HFClientVLLM

HFModel = dsp.HFModel
LlamaCpp = dsp.LlamaCpp

configure = settings.configure
context = settings.context
