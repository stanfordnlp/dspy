import dsp
from dsp.modules.hf_client import ChatModuleClient, HFClientSGLang, HFClientVLLM, HFServerTGI

from .predict import *
from .primitives import *
from .retrieve import *
from .signatures import *

import dspy.retrievers

# Functional must be imported after primitives, predict and signatures
from .functional import * # isort: skip
from dspy.evaluate import Evaluate # isort: skip
from dspy.clients import * # isort: skip
from dspy.adapters import * # isort: skip
from dspy.utils.logging_utils import configure_dspy_loggers, disable_logging, enable_logging
from dspy.utils.asyncify import asyncify

settings = dsp.settings

configure_dspy_loggers(__name__)

# LM = dsp.LM

AzureOpenAI = dsp.AzureOpenAI
OpenAI = dsp.GPT3
MultiOpenAI = dsp.MultiOpenAI
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
LlamaCpp = dsp.LlamaCpp

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


import dspy.teleprompt

LabeledFewShot = dspy.teleprompt.LabeledFewShot
BootstrapFewShot = dspy.teleprompt.BootstrapFewShot
BootstrapFewShotWithRandomSearch = dspy.teleprompt.BootstrapFewShotWithRandomSearch
BootstrapRS = dspy.teleprompt.BootstrapFewShotWithRandomSearch
BootstrapFinetune = dspy.teleprompt.BootstrapFinetune
BetterTogether = dspy.teleprompt.BetterTogether
COPRO = dspy.teleprompt.COPRO
MIPROv2 = dspy.teleprompt.MIPROv2
Ensemble = dspy.teleprompt.Ensemble
