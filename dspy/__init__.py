from dsp.modules.hf_client import ChatModuleClient
from dsp.modules.hf_client import HFServerTGI
from .signatures import *

from .retrieve import *
from .predict import *
from .primitives import *
# from .evaluation import *


# FIXME:


import dsp

settings = dsp.settings

OpenAI = dsp.GPT
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever

HFClientTGI = dsp.HFClientTGI

Anyscale = dsp.Anyscale
HFModel = dsp.HFModel
