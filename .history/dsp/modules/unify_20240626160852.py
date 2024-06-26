
import os 
import sys
import json
import requests
from typing import AsyncGenerator, Dict, Generator, List, Optional, Union
import openai
from unify.clients import Unify
from dsp.modules.lm import LM

class Unify(LM):
    def __init__(
        self, 
        endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03", 
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        api_key=None,
        **kwargs,  # Added to accept additional keyword arguments
    ):
        self.endpoint = endpoint
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        # self.api_provider: Literal["unify"] = "unify"
        self.api_base = "https://api.unify.ai/v0"
        self.model = endpoint
        super().__init__(model=self.model)
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 1,
            "top_k": 20,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "num_ctx": 1024,
            **kwargs,
        }  

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        """
        Send request to the Unify AI API.
        This method is required by the LM base class.
        """
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        
        settings_dict = {
            "model": self.model,
            "options": {k: v for k, v in kwargs.items() if k not in ["n", "max_tokens"]},
            "stream": False,
        }
        if self.model_type == "chat":
            settings_dict["messages"] = [{"role": "user", "content": prompt}]
        else:
            settings_dict["prompt"] = prompt
        
        # Call the generate method
        response = self._call_generate(settings_dict)
        return response

    def _call_generate(self, settings_dict):
        """
        Call the generate method from the unify client.
        """
        unify_instance = Unify()
        unify_instance.generate()
        
        try:
            response = unify_instance.generate(settings=settings_dict, api_key=self.api_key)
            return response
        except Exception as e:
            print(f"An error occurred while calling the generate method: {e}")
            return None