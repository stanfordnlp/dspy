import dsp
from dsp.modules.hf_client import HFClientVLLM

from .backends import *
from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *

# from .evaluation import *


# FIXME:


settings = dsp.settings

OpenAI = dsp.GPT3
Cohere = dsp.Cohere
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever
Clarifai = dsp.ClarifaiLLM

HFClientTGI = dsp.HFClientTGI
HFClientVLLM = HFClientVLLM

Anyscale = dsp.Anyscale
Together = dsp.Together
HFModel = dsp.HFModel
OllamaLocal = dsp.OllamaLocal
Bedrock = dsp.Bedrock

configure = settings.configure
context = settings.context
