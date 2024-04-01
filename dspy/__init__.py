import dsp
from dsp.modules.hf_client import ChatModuleClient, HFClientSGLang, HFClientVLLM, HFServerTGI

from .predict import *  # noqa: F403
from .primitives import *  # noqa: F403
from .retrieve import *  # noqa: F403
from .signatures import *  # noqa: F403
from .utils.logging import logger, set_log_level, set_log_output

# Functional must be imported after primitives, predict and signatures
from .functional import *  # noqa: F403   # isort: skip

settings = dsp.settings

AzureOpenAI = dsp.AzureOpenAI
OpenAI = dsp.GPT3
Mistral = dsp.Mistral
Databricks = dsp.Databricks
Cohere = dsp.Cohere
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever
Clarifai = dsp.ClarifaiLLM
Google = dsp.Google

HFClientTGI = dsp.HFClientTGI

Anyscale = dsp.Anyscale
Together = dsp.Together
HFModel = dsp.HFModel
OllamaLocal = dsp.OllamaLocal

Bedrock = dsp.Bedrock
Sagemaker = dsp.Sagemaker
AWSModel = dsp.AWSModel
AWSMistral = dsp.AWSMistral
AWSAnthropic = dsp.AWSAnthropic
AWSLlama2 = dsp.AWSLlama2

configure = settings.configure
context = settings.context
