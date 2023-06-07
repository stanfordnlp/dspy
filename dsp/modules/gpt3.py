import functools
import json
from typing import Any, Literal, Optional, cast

import backoff
import openai
import openai.error
from openai.openai_object import OpenAIObject
import dsp
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM

import itertools


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


class ChatAdapter:
    """
        OpenAI Turbo Docs. See https://platform.openai.com/docs/guides/chat/introduction
    """
    def __init__(self):
        self.prompt = {"model":"", "messages":[]}

    # def __call__(self, parts: list, include_rdemos: bool, include_ademos: bool, include_ademos: bool, include_context: bool):
    def __call__(self, parts: dict, multi_turn=False, assistant_turns=False):
        self.prompt["model"] = dsp.settings.lm.kwargs["model"]
        
        instructions, guidelines, rdemos, ademos, query, long_query = parts["instructions"], parts["guidelines"], parts["rdemos"], parts["ademos"], parts["query"], parts["long_query"]        
        rdemos = "\n\n".join(rdemos)
        messages = []
        
        if len(rdemos) >= 1 and len(ademos) == 0 and not long_query:
            # if not multi_turn:
            #     self.prompt["messages"] = [
            #         {"role": "user", "content": ("\n\n---\n\n".join([p.strip for k, p in parts.items()])).strip()}
            #     ]
            #     return self.prompt
            
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": guidelines},
                {"role": "user", "content": rdemos},
            ]
        elif len(rdemos) == 0:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": guidelines},
            ]
            for ademo in ademos:
                messages.append({"role": "user", "content": ademo})
            
        else:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": rdemos},
                {"role": "user", "content": guidelines},
            ]
            for ademo in ademos:
                messages.append({"role": "user", "content": ademo})
            
        messages.append({"role": "user", "content": query})
        self.prompt["messages"] = messages
        return self.prompt


class BasicAdapter:
    """
    Adapter Pattern. See https://www.geeksforgeeks.org/adapter-method-python-design-patterns/
    """
    def __init__(self):
        self.prompt = ""

    def __call__(self, parts: dict):
        instructions, guidelines, rdemos, ademos, query, long_query = parts["instructions"], parts["guidelines"], parts["rdemos"], parts["ademos"], parts["query"], parts["long_query"]
        rdemos = "\n\n".join(rdemos)
        if len(rdemos) >= 1 and len(ademos) == 0 and not long_query:
            rdemos_and_query = "\n\n".join([rdemos, query])
            parts = [
                instructions,
                guidelines,
                rdemos_and_query,
            ]
        elif len(rdemos) == 0:
            parts = [
                instructions,
                guidelines,
                *ademos,
                query,
            ]
        else:
            parts = [
                instructions,
                rdemos,
                guidelines,
                *ademos,
                query,
            ]

        self.prompt = "\n\n---\n\n".join([p.strip() for p in parts if p])
        self.prompt = self.prompt.strip()
        return self.prompt

class GPT3(LM):
    """Wrapper around OpenAI's GPT API. Supports both the OpenAI and Azure APIs.

    Args:
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "text-davinci-002".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "openai".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        model: str = "text-davinci-002",
        api_key: Optional[str] = None,
        api_provider: Literal["openai", "azure"] = "openai",
        model_type: Literal["chat", "text"] = "text",
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "openai"
        self.model_type = model_type
        self.adapter = ChatAdapter() if model_type=="chat" else BasicAdapter()

        if api_provider == "azure":
            assert (
                "engine" in kwargs or "deployment_id" in kwargs
            ), "Must specify engine or deployment_id for Azure API instead of model."
            assert "api_version" in kwargs, "Must specify api_version for Azure API"
            assert "api_base" in kwargs, "Must specify api_base for Azure API"
            openai.api_type = "azure"
            openai.api_base = kwargs["api_base"]
            if kwargs.get("api_version"):
                openai.api_version = kwargs["api_version"]

        if api_key:
            openai.api_key = api_key

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>
        if api_provider == "openai":
            self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt, **kwargs) -> OpenAIObject:
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":            
            kwargs["model"] = prompt["model"]
            kwargs["messages"] = tuple(prompt["messages"])
            kwargs = {
                "stringify_request": json.dumps(kwargs)
            }
            response = cached_gpt3_turbo_request(**kwargs)
        else:
            kwargs["prompt"] = prompt
            response = cached_gpt3_request(**kwargs)

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
        (openai.error.RateLimitError, openai.error.ServiceUnavailableError),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs) -> OpenAIObject:
        """Handles retreival of GPT-3 completions whilst handling rate limiting and caching."""
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

        if kwargs.get("n", 1) > 1:
            if self.model_type == "chat":
                kwargs = {**kwargs}
            else:
                kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)
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


@CacheMemory.cache
def cached_gpt3_request_v2(**kwargs):
    return openai.Completion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt3_request_v2_wrapped(**kwargs):
    return cached_gpt3_request_v2(**kwargs)


cached_gpt3_request = cached_gpt3_request_v2_wrapped


@CacheMemory.cache
def _cached_gpt3_turbo_request_v2(**kwargs) -> OpenAIObject:
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    return cast(OpenAIObject, openai.ChatCompletion.create(**kwargs))


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def _cached_gpt3_turbo_request_v2_wrapped(**kwargs) -> OpenAIObject:
    return _cached_gpt3_turbo_request_v2(**kwargs)


cached_gpt3_turbo_request = _cached_gpt3_turbo_request_v2_wrapped
