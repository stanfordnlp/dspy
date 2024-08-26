from typing import Any, Optional, Literal, Dict, List, Union, cast
from pydantic import BaseModel

import litellm
from litellm.types.utils import Choices, ModelResponse, Usage
from litellm.caching import Cache

import dspy
from dspy.utils.logging import logger

from dsp.modules.lm import LM
from dsp.utils.settings import settings

# Initialize the cache
litellm.cache = Cache(disk_cache_dir=".dspy_cache", type="disk")

# Use this for development testing
litellm.set_verbose = True

class Message(BaseModel):
    role: str
    content: Optional[str] = None # Only null for assistant messages with function calls
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class LLM(BaseModel):
    # TODO: See all commented out params and see if we need to support them.
    ### required params
    model: str
    model_type: str # This is needed for litellm.get_supported_openai_params call - this needs to be the name of the model itself (not the deployment name). Figure out a better name for this
    messages: List[Message] = []
    ### litellm-specific params
    api_key: Optional[str] = None 
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    num_retries: Optional[int] = 0
    # context_window_fallback_dict
    # fallbacks
    # metadata
    ### openai params
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 150
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    timeout: Optional[int] = 600
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0 # TODO: This might cause an error
    # stream
    # stream_options
    # response_format
    # tools
    # tool_choice
    # parallel_tool_calls
    # Note: litellm also allows a <provider>_key which is excluded as each LLM object points to a specific LLM
    ### Google - TODO: are these needed? Litellm docs confusing
    vertex_credentials: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    ### Custom prompt template TODO: see if needed https://docs.litellm.ai/docs/completion/input
    ### dspy params - these need to be removed from litellm calls
    history: Optional[List[Dict[str, Any]]] = None
    provider: Optional[str] = "openai"

    # Allows extra kwargs to be passed in
    class Config:
        extra = "allow"

    def request(self, **kwargs):
        """
        Litellm request
        """
        pass

    def required_params(self):
        return 

    def set_provider(self):
        # TODO: confirm if it's ok to assume the provider is openai or the prefix as https://docs.litellm.ai/docs/providers/aleph_alpha doesn't seem to include a prefix
        if "/" in self.model:
            self.provider = self.model.split("/")[0]

    def pre_process(self, drop_params, **kwargs):   
        # Add caching to kwargs
        enable_cache = dspy.settings.enable_cache
        if "enable_cache" in kwargs: 
            # request param takes precedence
            enable_cache = kwargs["enable_cache"]
        if enable_cache: 
            kwargs['cache'] = {"no-cache": False, "no-store": False}
        else:
            kwargs['cache'] = {"no-cache": True, "no-store": True}

        # get litellm params and exclude rest
        if drop_params:
            kwargs["drop_params"] = True
        
        # set the provider
        self.set_provider()

        supported_params = litellm.get_supported_openai_params(model=self.model_type, custom_llm_provider=self.provider)




    def __call__(
        self, 
        prompt: str, 
        only_completed: bool = False,
        return_sorted: bool = False,
        drop_params: bool = True,
        additional_drop_params: list = [],
        **kwargs
    ) -> list[dict[str, Any]]:
        
        self.pre_process(drop_params, **kwargs)

        response = self.request(**kwargs)

        # postprocessing


"""
Useful litellm helper functions/flags:

check_valid_key() -> Check if a user submitted a valid key for the model they're trying to call.

litellm.get_supported_openai_params() -> get list of params for each model + provider

litellm.drop_params = True
"""