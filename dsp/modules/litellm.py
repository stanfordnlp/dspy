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
    content: Optional[str] # Only null for assistant messages with function calls
    name: Optional[str]
    function_call: Optional[Dict[str, Any]]

    def __init__(self, **data):  
        super().__init__(**data)  
        self.__dict__ = {  
            k: v for k, v in self.__dict__.items() if v is not None  
        }  

    # TODO: test this
    @classmethod  
    def create_from_kwargs(cls, kwargs_list):  
        kwargs_dict = {}  
        for kwargs in kwargs_list:  
            kwargs_dict.update(kwargs)  
        return cls(**kwargs_dict)
    
    def to_dict(self):  
        kwargs = self.model_dump(exclude_unset=True)
        return kwargs

class LLM(BaseModel):
    ### required params
    model: str
    # model_type: str # This is needed for litellm.get_supported_openai_params call - this needs to be the name of the model itself (not the deployment name).
    messages: Optional[List[Message]] = []
    ### dspy params 
    history: Optional[List[Dict[str, Any]]] = []
    provider: Optional[str] = "openai" # TODO: do we need the provider?
    # These need to be removed from litellm calls TODO: this list isn't needed
    # non_litellm_params: List[str] = ["model_type", "history", "provider"]

    # Allows extra kwargs to be passed in
    class Config:
        extra = "allow"

    def request(self, **kwargs):
        """
        Litellm request
        """
        print(kwargs)

        response = litellm.completion(**kwargs)
        return response

    def update_messages(self, input, input_type):
        if input_type == "prompt":
            self.messages.append(Message(role="user", content=input, name=None, function_call=None))
        else: #list of messages
            for msg in input:
                self.messages.append(Message.create_from_kwargs(msg))
        return

    def set_provider(self):
        # TODO: confirm if it's ok to assume the provider is openai or the prefix as https://docs.litellm.ai/docs/providers/aleph_alpha doesn't seem to include a prefix
        # TODO: might not be needed unless we want to use litellm.get_supported_openai_params() to get list of params for each model + provider and validate all the args before the llm call
        if "/" in self.model:
            self.provider = self.model.split("/")[0]
        
    def postprocess(self, litellm_response, input_type, only_completed, org_kwargs, litellm_kwargs):

        # extract litellm response

        # to standardize the history, the input is always in the form of the message sent to litellm, however input_type indicates whether it was passed in as a prompt or as is. TODO: review & confirm.

        org_kwargs["only_completed"] = only_completed
        # drop_params already in litellm_kwargs

        # This could technically be a update_history function
        # TODO: I changed the fields in the history, does that impact anything?
        history = {
            "input": [msg.to_dict() for msg in self.messages],
            "input_type": input_type,
            "response": litellm_response,
            "kwargs": org_kwargs,
            "litellm_kwargs": litellm_kwargs,
        }

        self.history.append(history)
        self.messages = [] # TODO: change this if we want to accumlate message history for litellm

        result = []

        if only_completed:
            for choice in litellm_response.choices: 
                if choice.finish_reason == "stop":
                    result.append(choice)
            return result
        else:
            return litellm_response.choices


    def preprocess(self, drop_params, input, input_type, **kwargs):   
        """
        Update the kwargs with the necessary parameters to make the litellm call
        """
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

        # add the litellm params to kwargs
        kwargs["model"] = self.model

        # TODO: see notes in function def
        self.set_provider()

        self.update_messages(input, input_type) # TODO: should we always send all the messages to the litellm call? Or do we need to have a parameter for "keep n messages?" Note that it's not as simple as just truncating the start of the list if we're considering the system message, which would need to stay regardless of the truncation.

        kwargs["messages"] = []
        for msg in self.messages:
            kwargs["messages"].append(msg.to_dict())

        return kwargs

    def __call__(
        self, 
        prompt: Optional[str] = None,
        messages = None, #providing this as an option if someone wants to pass in >1 messsages, especially if they want to include a system message. TODO: better way to do it? Does it make sense to allow a system message in init?
        only_completed: bool = False,
        drop_params: bool = True,
        **kwargs
    ) -> list[dict[str, Any]]:
        
        assert (prompt is not None) != (messages is not None), "Either 'prompt' or 'message' must be set, but not both."

        if prompt: 
            input = prompt
            input_type = "prompt"
        else: 
            input = messages
            input_type = "message"

        # all_kwargs = self.model_dump()   
          
        # # Filter out the standard fields to get only the extra kwargs  
        # standard_fields = set(self.__fields__.keys())  
        # extra_kwargs = {k: v for k, v in all_kwargs.items() if k not in standard_fields}
        extra_kwargs = self.model_extra
  
        # Merge the kwargs with any additional ones passed directly to the method
        kwargs = kwargs | extra_kwargs  

        litellm_kwargs = self.preprocess(drop_params, input, input_type, **kwargs) 
        litellm_response = self.request(**litellm_kwargs)
        result = self.postprocess(litellm_response, input_type, only_completed, kwargs, litellm_kwargs) 

        return result