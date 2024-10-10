import functools
import json
import logging
from typing import Any, Callable, Literal, Optional, cast

import backoff

try:
    """
    If there is any error in the langfuse configuration, it will turn to request the real address(openai or azure endpoint)
    """
    import langfuse
    from langfuse.openai import openai
    logging.info(f"You are using Langfuse,version{langfuse.__version__}")
except:
    import openai

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM
from dsp.utils.settings import settings


try:
    OPENAI_LEGACY = int(openai.version.__version__[0]) == 0
except Exception:
    OPENAI_LEGACY = True

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


AzureADTokenProvider = Callable[[], str]


class AzureOpenAI(LM):
    """Wrapper around Azure's API for OpenAI.

    Args:
        api_base (str): Azure URL endpoint for model calling, often called 'azure_endpoint'.
        api_version (str): Version identifier for API.
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "gpt-3.5-turbo-instruct".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "chat".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        api_base: str,
        api_version: str,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "openai"

        self.system_prompt = system_prompt

        # Define Client
        if OPENAI_LEGACY:
            # Assert that all variables are available
            assert (
                "engine" in kwargs or "deployment_id" in kwargs
            ), "Must specify engine or deployment_id for Azure API instead of model."

            openai.api_base = api_base
            openai.api_key = api_key
            openai.api_type = "azure"
            openai.api_version = api_version
            openai.azure_ad_token_provider = azure_ad_token_provider

            self.client = None

        else:
            client = openai.AzureOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                api_version=api_version,
                azure_ad_token_provider=azure_ad_token_provider,
            )

            self.client = client

        self.model_type = model_type

        if not OPENAI_LEGACY and "model" not in kwargs:
            if "deployment_id" in kwargs:
                kwargs["model"] = kwargs["deployment_id"]
                del kwargs["deployment_id"]

            if "api_version" in kwargs:
                del kwargs["api_version"]

        if "model" not in kwargs:
            kwargs["model"] = model

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key

        self.history: list[dict[str, Any]] = []

    def _openai_client(self):
        if OPENAI_LEGACY:
            return openai

        return self.client

    def log_usage(self, response):
        """Log the total tokens from the Azure OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.debug(f"Azure OpenAI Total Token Usage: {total_tokens}")

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            kwargs["messages"] = messages
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = chat_request(self.client, **kwargs)

        else:
            kwargs["prompt"] = prompt
            response = completions_request(self.client, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of GPT-3 completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == "chat":
            return choice["message"]["content"]
        return choice["text"]

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from OpenAI Model.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.request(prompt, **kwargs)

        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(
            model=model,
            api_key=self.api_key,
            api_version=self.api_version,
            api_base=self.api_base,
            **kwargs,
        )


@CacheMemory.cache
def cached_gpt3_request_v2(**kwargs):
    return openai.Completion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt3_request_v2_wrapped(**kwargs):
    return cached_gpt3_request_v2(**kwargs)


@CacheMemory.cache
def _cached_gpt3_turbo_request_v2(**kwargs) -> OpenAIObject:
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    return cast(OpenAIObject, openai.ChatCompletion.create(**kwargs))


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def _cached_gpt3_turbo_request_v2_wrapped(**kwargs) -> OpenAIObject:
    return _cached_gpt3_turbo_request_v2(**kwargs)


def v1_chat_request(client, **kwargs):
    @functools.lru_cache(maxsize=None if cache_turn_on else 0)
    @NotebookCacheMemory.cache
    def v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs):
        @CacheMemory.cache
        def v1_cached_gpt3_turbo_request_v2(**kwargs):
            if "stringify_request" in kwargs:
                kwargs = json.loads(kwargs["stringify_request"])
            return client.chat.completions.create(**kwargs)

        return v1_cached_gpt3_turbo_request_v2(**kwargs)

    return v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs).model_dump()


def v1_completions_request(client, **kwargs):
    @functools.lru_cache(maxsize=None if cache_turn_on else 0)
    @NotebookCacheMemory.cache
    def v1_cached_gpt3_request_v2_wrapped(**kwargs):
        @CacheMemory.cache
        def v1_cached_gpt3_request_v2(**kwargs):
            return client.completions.create(**kwargs)

        return v1_cached_gpt3_request_v2(**kwargs)

    return v1_cached_gpt3_request_v2_wrapped(**kwargs).model_dump()


def chat_request(client, **kwargs):
    if OPENAI_LEGACY:
        return _cached_gpt3_turbo_request_v2_wrapped(**kwargs)

    return v1_chat_request(client, **kwargs)


def completions_request(client, **kwargs):
    if OPENAI_LEGACY:
        return cached_gpt3_request_v2_wrapped(**kwargs)

    return v1_completions_request(client, **kwargs)
