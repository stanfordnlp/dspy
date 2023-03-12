import functools
import json
from typing import Optional, Any
import openai
import openai.error
from openai.openai_object import OpenAIObject
import backoff

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


class GPT3:
    """Wrapper around OpenAI's GPT-3 API."""

    def __init__(
        self, model: str = "text-davinci-002", api_key: Optional[str] = None, **kwargs
    ):
        if api_key:
            openai.api_key = api_key

        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.history: list[dict[str, Any]] = []

    def _basic_request(self, prompt: str, **kwargs) -> OpenAIObject:
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if kwargs["model"] in ("gpt-3.5-turbo", "gpt-3.5-turbo-0301"):
            kwargs["messages"] = json.dumps([{"role": "user", "content": prompt}])
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
        return self._basic_request(prompt, **kwargs)

    def print_green(self, text: str, end: str = "\n"):
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)

    def print_red(self, text: str, end: str = "\n"):
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)

    def inspect_history(self, n: int = 1):
        """Prints the last n prompts and their completions.
        TODO: print the valid choice that contains filled output field instead of the first
        """
        last_prompt = None
        printed = []

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
                printed.append((prompt, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        for prompt, choices in reversed(printed):
            print("\n\n\n")
            print(prompt, end="")
            self.print_green(self._get_choice_text(choices[0]), end="")
            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices)-1} other completions)", end="")
            print("\n\n\n")

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.kwargs["model"] in ("gpt-3.5-turbo", "gpt-3.5-turbo-0301"):
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
            if self.kwargs["model"] in ("gpt-3.5-turbo", "gpt-3.5-turbo-0301"):
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
def cached_gpt3_turbo_request_v2(**kwargs):
    kwargs["messages"] = json.loads(kwargs["messages"])
    return openai.ChatCompletion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt3_turbo_request_v2_wrapped(**kwargs):
    return cached_gpt3_turbo_request_v2(**kwargs)


cached_gpt3_turbo_request = cached_gpt3_turbo_request_v2_wrapped
