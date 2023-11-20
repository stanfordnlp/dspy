from dsp.modules.hf_client import ChatModuleClient
from dsp.modules.hf_client import HFServerTGI, HFClientVLLM
from .signatures import *

from .retrieve import *
from .predict import *
from .primitives import *
# from .evaluation import *


# FIXME:


import dsp

settings = dsp.settings

OpenAI = dsp.GPT3
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever

HFClientTGI = dsp.HFClientTGI
HFClientVLLM = HFClientVLLM

Anyscale = dsp.Anyscale
HFModel = dsp.HFModel
