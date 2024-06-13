import functools
import json
from typing import Literal, Optional

import openai

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.gpt3 import GPT3

try:
    import openai.error
    from openai.openai_object import OpenAIObject
    ERRORS = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError)
except Exception:
    ERRORS = (openai.RateLimitError, openai.APIError)
    OpenAIObject = dict


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )

class Databricks(GPT3):
    """Wrapper around DSPy's OpenAI Wrapper. Supports Databricks Model Serving Endpoints for OpenAI SDK on both Chat, Completions, and Embeddings models.

    Args:
        model (str, required): Databricks-hosted LLM model to use.
        api_key (Optional[str], optional): Databricks authentication token. Defaults to None.
        api_base (Optional[str], optional): Databricks model serving endpoint. Defaults to None.
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the OpenAI API provider.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_type: Literal["chat", "text", "embeddings"] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            api_provider="openai",
            api_base=api_base,
            model_type=model_type,
            **kwargs,
        )

        self.kwargs.pop('frequency_penalty', None)
        self.kwargs.pop('presence_penalty', None)    

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            kwargs["messages"] = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = custom_client_chat_request(**kwargs).json()
            response = json.loads(response)
        else:
            kwargs["prompt"] = prompt
            response = custom_client_completions_request(**kwargs).json()    
            response = json.loads(response)
        
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        return response
    
    def embeddings(self, prompt: str, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        kwargs["input"] = prompt
        kwargs.pop('temperature', None)
        kwargs.pop('max_tokens', None)
        kwargs.pop('top_p', None)
        kwargs.pop('n', None)
        response = custom_client_embeddings_request(**kwargs).json()    
        response = json.loads(response)
        embeddings = [cur_obj['embedding'] for cur_obj in response['data']][0]
        return embeddings
    
    def __call__(self, prompt: str, **kwargs):
        if self.model_type == "embeddings":
            return self.embeddings(prompt, **kwargs)
        else:
            return super().__call__(prompt, **kwargs)

def create_custom_client():
    from openai import OpenAI
    
    client = OpenAI(api_key=openai.api_key, base_url=openai.base_url)
    return client

def custom_client_chat_request(**kwargs):
    return cached_custom_client_chat_request_v2_wrapped(**kwargs)

def custom_client_embeddings_request(**kwargs):
    return cached_custom_client_embeddings_request_v2_wrapped(**kwargs)

def custom_client_completions_request(**kwargs):
    return cached_custom_client_completions_request_v2_wrapped(**kwargs)


@CacheMemory.cache
def cached_custom_client_chat_request_v2(**kwargs):
    client = create_custom_client()
    return client.chat.completions.create(**kwargs)

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_custom_client_chat_request_v2_wrapped(**kwargs):
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    return cached_custom_client_chat_request_v2(**kwargs)

@CacheMemory.cache
def cached_custom_client_completions_request_v2(**kwargs):
    client = create_custom_client()
    return client.completions.create(**kwargs)

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_custom_client_completions_request_v2_wrapped(**kwargs):
    return cached_custom_client_completions_request_v2(**kwargs)

@CacheMemory.cache
def cached_custom_client_embeddings_request_v2(**kwargs):
    client = create_custom_client()
    return client.embeddings.create(**kwargs)

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_custom_client_embeddings_request_v2_wrapped(**kwargs):
    return cached_custom_client_embeddings_request_v2(**kwargs)
