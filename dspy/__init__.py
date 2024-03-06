import dsp
from dsp.modules.hf_client import ChatModuleClient, HFClientSGLang, HFClientVLLM, HFServerTGI

from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *

# Functional must be imported after primitives, predict and signatures
from .functional import * # isort: skip

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
