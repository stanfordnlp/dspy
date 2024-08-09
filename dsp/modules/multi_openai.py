import functools
import json
import logging
from typing import Any, Literal, Optional, cast

import backoff
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

    ERRORS = (openai.error.RateLimitError,)
except Exception:
    ERRORS = (openai.RateLimitError,)
    OpenAIObject = dict


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class MultiOpenAI(LM):
    """Wrapper around OpenAI Compatible API.

    Args:
        model (str): LLM model to use.
        api_key (Optional[str]): API provider Authentication token.
        api_provider (str): The API provider to use.
        model_type (Literal["chat", "text"]): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "chat".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        api_provider: str,
        api_base: str,
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = api_provider
        self.model_type = model_type

        self.system_prompt = system_prompt

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }

        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

        if OPENAI_LEGACY:
            openai.api_base = api_base

            if api_key:
                openai.api_key = api_key

            def legacy_chat_request_wrapped(**kwargs):
                @functools.lru_cache(maxsize=None if cache_turn_on else 0)
                @NotebookCacheMemory.cache
                def cached_legacy_chat_request_wrapped(**kwargs):
                    @CacheMemory.cache
                    def cached_legacy_chat_request(**kwargs):
                        if "stringify_request" in kwargs:
                            kwargs = json.loads(kwargs["stringify_request"])
                        return cast(OpenAIObject, openai.ChatCompletion.create(**kwargs))

                    return cached_legacy_chat_request(**kwargs)

                return cached_legacy_chat_request_wrapped(**kwargs)

            def legacy_completions_request_wrapped(**kwargs):
                @functools.lru_cache(maxsize=None if cache_turn_on else 0)
                @NotebookCacheMemory.cache
                def cached_legacy_completions_request_wrapped(**kwargs):
                    @CacheMemory.cache
                    def cached_legacy_completions_request(**kwargs):
                        return openai.Completion.create(**kwargs)

                    return cached_legacy_completions_request(**kwargs)

                return cached_legacy_completions_request_wrapped(**kwargs)

            self.chat_request = legacy_chat_request_wrapped
            self.completions_request = legacy_completions_request_wrapped

        else:
            from openai import OpenAI
            openai_client = OpenAI(api_key=api_key, base_url=api_base)

            def chat_request_wrapped(**kwargs):
                @functools.lru_cache(maxsize=None if cache_turn_on else 0)
                @NotebookCacheMemory.cache
                def cached_chat_request_wrapped(**kwargs):
                    @CacheMemory.cache
                    def cached_chat_request(**kwargs):
                        if "stringify_request" in kwargs:
                            kwargs = json.loads(kwargs["stringify_request"])
                        return openai_client.chat.completions.create(**kwargs)

                    return cached_chat_request(**kwargs)

                return cached_chat_request_wrapped(**kwargs).model_dump()

            def completions_request_wrapped(**kwargs):
                @functools.lru_cache(maxsize=None if cache_turn_on else 0)
                @NotebookCacheMemory.cache
                def cached_completions_request_wrapped(**kwargs):
                    @CacheMemory.cache
                    def cached_completions_request(**kwargs):
                        return openai_client.completions.create(**kwargs)

                    return cached_completions_request(**kwargs)

                return cached_completions_request_wrapped(**kwargs).model_dump()

            self.chat_request = chat_request_wrapped
            self.completions_request = completions_request_wrapped

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.debug(f"OpenAI Response Token Usage: {total_tokens}")

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
            response = self.chat_request(**kwargs)

        else:
            kwargs["prompt"] = prompt
            response = self.completions_request(**kwargs)

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
        """Retrieves completions from GPT-3.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        self.log_usage(response)
        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        if kwargs.get("logprobs", False):
            completions = [{'text': self._get_choice_text(c), 'logprobs': c["logprobs"]} for c in choices]
        else:
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
                scored_completions.append((avglog, self._get_choice_text(c), logprobs))
            scored_completions = sorted(scored_completions, reverse=True)
            if logprobs:
                completions = [{'text': c, 'logprobs': lp} for _, c, lp in scored_completions]
            else:
                completions = [c for _, c in scored_completions]

        return completions

