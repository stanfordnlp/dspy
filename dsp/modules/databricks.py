import json
import functools
from typing import Literal, Optional


from dsp.modules.cache_utils import (
    CacheMemory,
    NotebookCacheMemory,
    cache_turn_on,
)

try:
    import openai.error
    from openai.openai_object import OpenAIObject

    ERRORS = (
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
    )
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


class Databricks:
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
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.api_base = api_base
        self.api_provider = "openai"
        self.api_key = api_key
        self.kwargs = kwargs or {}
        self.kwargs = {
            **self.kwargs,
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
        }

    def embeddings(self, prompt: str, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        kwargs["input"] = prompt

        response = custom_client_embeddings_request(**kwargs).json()
        response = json.loads(response)
        embeddings = [cur_obj["embedding"] for cur_obj in response["data"]][0]
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


def custom_client_embeddings_request(**kwargs):
    return cached_custom_client_embeddings_request_v2_wrapped(**kwargs)


@CacheMemory.cache
def cached_custom_client_embeddings_request_v2(**kwargs):
    client = create_custom_client()
    return client.embeddings.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_custom_client_embeddings_request_v2_wrapped(**kwargs):
    return cached_custom_client_embeddings_request_v2(**kwargs)
