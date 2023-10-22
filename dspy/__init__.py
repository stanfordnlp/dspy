from typing import Optional
from dsp.modules.hf_client import ChatModuleClient
from dspy.callbacks.base_handler import BaseCallbackHandler
from dspy.callbacks.global_handlers import set_global_handler
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
ChatModuleClient = ChatModuleClient
HFModel = dsp.HFModel
global_handler: Optional[BaseCallbackHandler] = None