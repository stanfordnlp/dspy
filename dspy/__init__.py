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

OpenAI = dsp.GPT3
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever

HFClientTGI = dsp.HFClientTGI
HFClientVLLM = HFClientVLLM

Anyscale = dsp.Anyscale
HFModel = dsp.HFModel

configure = settings.configure
context = settings.context


import json
import os
import openai

class Models:
    """ Singleton to easily create LMs and keep track of all LMs throughout DSPy."""
    _instance = None
    lms = {}

    def __new__(cls, config_path=None):
        """Singleton constructor. Optionally takes a path to a config which is uses to instantiate LMs."""
        if not cls._instance:
            cls._instance = super(Models, cls).__new__(cls)
            cls._instance._configure_dspy(config_path=config_path)
        return cls._instance

    @classmethod
    def create_lm(cls, model_config):
        """ Create an LM based."""
        try:
            model_config = {**model_config} # deepcopy config
            model_name = model_config.pop('model')

            if "gpt-" in model_name:
                lm = OpenAI(model=model_name, **model_config)
            else:
                url = model_config.pop('url')
                lm = HFClientTGI(
                    model=model_name, url=url, port=[7140, 7141, 7142, 7143], **model_config
                )
            return lm
        except Exception as error:
            print(error)
            return None


    @classmethod
    def get_lm(cls, config_name):
        return cls._instance.lms[config_name]

    def _configure_dspy(self, config_path=None):
        """Read in an LM config, create LMs and save them."""
        # set openai key
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        # get LM config
        self.config_path = config_path

        if self.config_path:
            with open(self.config_path, "r") as fp:
                self.config = json.load(fp)

            for config_name, model_config in self.config.items():
                lm = self.create_lm(model_config)
                self.lms.update({config_name: lm})
