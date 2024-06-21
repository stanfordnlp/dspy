import dsp
from dsp.modules.hf_client import ChatModuleClient, HFClientSGLang, HFClientVLLM, HFServerTGI

from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *
from .utils.logging import logger, set_log_output

# Functional must be imported after primitives, predict and signatures
from .functional import *  # isort: skip

settings = dsp.settings

LM = dsp.LM

AzureOpenAI = dsp.AzureOpenAI
OpenAI = dsp.GPT3
Mistral = dsp.Mistral
Databricks = dsp.Databricks
Cohere = dsp.Cohere
ColBERTv2 = dsp.ColBERTv2
ColBERTv2RerankerLocal = dsp.ColBERTv2RerankerLocal
ColBERTv2RetrieverLocal = dsp.ColBERTv2RetrieverLocal
Pyserini = dsp.PyseriniRetriever
Clarifai = dsp.ClarifaiLLM
CloudflareAI = dsp.CloudflareAI
Google = dsp.Google
GoogleVertexAI = dsp.GoogleVertexAI
GROQ = dsp.GroqLM
Snowflake = dsp.Snowflake
Claude = dsp.Claude

HFClientTGI = dsp.HFClientTGI
HFClientVLLM = HFClientVLLM

Anyscale = dsp.Anyscale
Together = dsp.Together
HFModel = dsp.HFModel
OllamaLocal = dsp.OllamaLocal

Bedrock = dsp.Bedrock
Sagemaker = dsp.Sagemaker
AWSModel = dsp.AWSModel
AWSMistral = dsp.AWSMistral
AWSAnthropic = dsp.AWSAnthropic
AWSMeta = dsp.AWSMeta

Watsonx = dsp.Watsonx
PremAI = dsp.PremAI

You = dsp.You

configure = settings.configure
context = settings.context
