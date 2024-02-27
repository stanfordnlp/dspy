from dsp.modules.hf_client import ChatModuleClient
from dsp.modules.hf_client import HFServerTGI, HFClientVLLM, HFClientSGLang
from .signatures import *

from .retrieve import *
from .predict import *
from .primitives import *

# from .evaluation import *


# FIXME:


import dsp

settings = dsp.settings

AzureOpenAI = dsp.AzureOpenAI
OpenAI = dsp.GPT3
Databricks = dsp.Databricks
Cohere = dsp.Cohere
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever
Clarifai = dsp.ClarifaiLLM
Google = dsp.Google

HFClientTGI = dsp.HFClientTGI
HFClientVLLM = HFClientVLLM

Anyscale = dsp.Anyscale
Together = dsp.Together
HFModel = dsp.HFModel
OllamaLocal = dsp.OllamaLocal
Bedrock = dsp.Bedrock

configure = settings.configure
context = settings.context
