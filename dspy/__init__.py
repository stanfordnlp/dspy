from dsp.modules.hf_client import ChatModuleClient
from .signatures import *

from .retrieve import *
from .predict import *
from .primitives import *
from .assertions import *
# from .evaluation import *


# FIXME:


import dsp

settings = dsp.settings

OpenAI = dsp.GPT3
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever
HFClientTGI = dsp.HFClientTGI
ChatModuleClient = ChatModuleClient

print("Running from local DSPy package.")