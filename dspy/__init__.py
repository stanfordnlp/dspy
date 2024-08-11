import dsp
from dsp.modules.hf_client import (
    ChatModuleClient,
    HFClientSGLang,
    HFClientVLLM,
    HFServerTGI,
)

from dsp.modules.schemas import (
    ChatMessage,
    DSPyModelResponse,
    LLMParams,
)

from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *
from .utils.logging import logger, set_log_output

# Functional must be imported after primitives, predict and signatures
from .functional import *  # isort: skip

settings = dsp.settings

LM = dsp.LM

LLM = dsp.LLM

ColBERTv2 = dsp.ColBERTv2
ColBERTv2RerankerLocal = dsp.ColBERTv2RerankerLocal
ColBERTv2RetrieverLocal = dsp.ColBERTv2RetrieverLocal
Databricks = dsp.Databricks  # for embeddings call
Pyserini = dsp.PyseriniRetriever
CloudflareAI = dsp.CloudflareAI

HFClientTGI = dsp.HFClientTGI
HFClientVLLM = HFClientVLLM

Anyscale = dsp.Anyscale
Together = dsp.Together
HFModel = dsp.HFModel
LlamaCpp = dsp.LlamaCpp

Bedrock = dsp.Bedrock
Sagemaker = dsp.Sagemaker
AWSModel = dsp.AWSModel
AWSMistral = dsp.AWSMistral
AWSAnthropic = dsp.AWSAnthropic
AWSMeta = dsp.AWSMeta

configure = settings.configure
context = settings.context
