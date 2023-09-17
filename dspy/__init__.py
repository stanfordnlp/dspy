from internals.modules.hf_client import ChatModuleClient
from .signatures import *

from .retrieve import *
from .predict import *
from .primitives import *
# from .evaluation import *


# FIXME:


import internals

settings = internals.settings

OpenAI = internals.GPT3
ColBERTv2 = internals.ColBERTv2
Pyserini = internals.PyseriniRetriever
HFClientTGI = internals.HFClientTGI
ChatModuleClient = ChatModuleClient