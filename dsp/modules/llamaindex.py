"""
this code implements a wrapper around the llama_index library to emulate a dspy llm

this allows the llama_index library to be used in the dspy framework since dspy has limited support for LLMs

This code is a slightly modified copy of dspy/dsp/modules/azure_openai.py

The way this works is simply by creating a dummy openai client that wraps around any llama_index LLM object and implements .complete and .chat

tested with python 3.12

dspy==0.1.4
dspy-ai==2.4.9
llama-index==0.10.35
llama-index-llms-openai==0.1.18

"""

import json
import logging
from typing import Any, Literal

from easydict import EasyDict
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM


def LlamaIndexOpenAIClientWrapper(llm: LLM):
    def chat(messages: list[ChatMessage], **kwargs) -> Any:
        return llm.chat([ChatMessage(**message) for message in messages], **kwargs)

    def complete(prompt: str, **kwargs) -> Any:
        return llm.complete(prompt, **kwargs)

    client = EasyDict(
        {
            'chat': EasyDict({'completions': EasyDict({'create': chat})}),
            'completion': EasyDict({'create': complete}),
            'ChatCompletion': EasyDict({'create': chat}),
            'Completion': EasyDict({'create': complete}),
        }
    )
    return client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler('azure_openai_usage.log')],
)

import functools
import json
from typing import Any, Literal

import backoff
import dsp
import openai
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM

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
        'Backing off {wait:0.1f} seconds after {tries} tries ' 'calling function {target} with kwargs ' '{kwargs}'.format(**details),
    )


class DspyLlamaIndexWrapper(LM):
    """Wrapper around Azure's API for OpenAI.

    Args:
        api_base (str): Azure URL endpoint for model calling, often called 'azure_endpoint'.
        api_version (str): Version identifier for API.
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "text-davinci-002".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "chat".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        llm: LLM,
        model_type: Literal['chat', 'text'] = 'chat',
        **kwargs,
    ):
        super().__init__(llm.model)
        self.provider = 'openai'

        self.llm = llm
        self.client = LlamaIndexOpenAIClientWrapper(llm)
        model = llm.model
        self.model_type = model_type

        # if not OPENAI_LEGACY and "model" not in kwargs:
        #     if "deployment_id" in kwargs:
        #         kwargs["model"] = kwargs["deployment_id"]
        #         del kwargs["deployment_id"]

        #     if "api_version" in kwargs:
        #         del kwargs["api_version"]

        if 'model' not in kwargs:
            kwargs['model'] = model

        self.kwargs = {
            'temperature': 0.0,
            'max_tokens': 150,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'n': 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.history: list[dict[str, Any]] = []

    def _openai_client(self):
        # if OPENAI_LEGACY:
        #     return openai

        return self.client

    def log_usage(self, response):
        """Log the total tokens from the Azure OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            total_tokens = usage_data.get('total_tokens')
            logging.info(f'{total_tokens}')

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == 'chat':
            # caching mechanism requires hashable kwargs
            kwargs['messages'] = [{'role': 'user', 'content': prompt}]
            kwargs = {'stringify_request': json.dumps(kwargs)}
            # response = chat_request(self.client, **kwargs)
            # if OPENAI_LEGACY:
            #     return _cached_gpt3_turbo_request_v2_wrapped(**kwargs)
            # else:
            return v1_chat_request(self.client, **kwargs)

        else:
            kwargs['prompt'] = prompt
            response = self.completions_request(**kwargs)

        history = {
            'prompt': prompt,
            'response': response,
            'kwargs': kwargs,
            'raw_kwargs': raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of GPT-3 completions whilst handling rate limiting and caching."""
        if 'model_type' in kwargs:
            del kwargs['model_type']

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == 'chat':
            return choice['message']['content']
        return choice['text']

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

        assert only_completed, 'for now'
        assert return_sorted is False, 'for now'

        response = self.request(prompt, **kwargs)

        try:
            if dsp.settings.log_openai_usage:
                self.log_usage(response)
        except Exception:
            pass

        choices = response['choices']

        completed_choices = [c for c in choices if c['finish_reason'] != 'length']

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get('n', 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c['logprobs']['tokens'],
                    c['logprobs']['token_logprobs'],
                )

                if '<|endoftext|>' in tokens:
                    index = tokens.index('<|endoftext|>') + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions

    def completions_request(self, **kwargs):
        # if OPENAI_LEGACY:
        #     return cached_gpt3_request_v2_wrapped(**kwargs)
        return v1_completions_request(self.client, **kwargs)


def v1_chat_request(client, **kwargs):
    @functools.lru_cache(maxsize=None if cache_turn_on else 0)
    @NotebookCacheMemory.cache
    def v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs):
        @CacheMemory.cache
        def v1_cached_gpt3_turbo_request_v2(**kwargs):
            if 'stringify_request' in kwargs:
                kwargs = json.loads(kwargs['stringify_request'])
            return client.chat.completions.create(**kwargs)

        return v1_cached_gpt3_turbo_request_v2(**kwargs)

    response = v1_cached_gpt3_turbo_request_v2_wrapped(**kwargs)

    try:
        response = response.model_dump()
    except Exception:
        response = response.raw
        response['choices'] = [json.loads(x.json()) for x in response['choices']]
        response['usage'] = json.loads(response['usage'].json())
    return response


def v1_completions_request(client, **kwargs):
    @functools.lru_cache(maxsize=None if cache_turn_on else 0)
    @NotebookCacheMemory.cache
    def v1_cached_gpt3_request_v2_wrapped(**kwargs):
        @CacheMemory.cache
        def v1_cached_gpt3_request_v2(**kwargs):
            return client.completions.create(**kwargs)

        return v1_cached_gpt3_request_v2(**kwargs)

    return v1_cached_gpt3_request_v2_wrapped(**kwargs).model_dump()


## ======== test =========

if __name__ == '__main__':
    print('Testing DspyLlamaIndexWrapper')
    import os

    import dspy
    from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'], model='gpt-3.5-turbo')
    dspy_llm = DspyLlamaIndexWrapper(llm)

    # Load math questions from the GSM8K dataset.
    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought('question -> answer')

        def forward(self, question):
            response = self.prog(question=question)
            return response

    ##

    dspy.settings.configure(lm=dspy_llm)

    from dspy.teleprompt import BootstrapFewShot

    # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

    # Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)
    print(f'{optimized_cot=}')
