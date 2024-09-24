import queue
import threading
from threading import Lock, Event, Thread
from dsp.modules.lm import LM
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.utils.settings import settings
from vllm.entrypoints.chat_utils import parse_chat_messages

import uuid
import backoff
from typing import Any, Generator, List, Callable, Awaitable, AsyncGenerator, Generic, TypeVar
import functools
import asyncio
import contextlib
import random
import collections
import time
import ujson
from dsp.deploy_dspy.download import download_model
from dsp.deploy_dspy.async_llm import AsyncLLMWrapper
from transformers import AutoTokenizer
import msgspec
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs, RequestOutput
from vllm.lora.request import LoRARequest
from asyncio import FIRST_COMPLETED, ensure_future
try:
    from vllm.entrypoints.chat_utils import apply_chat_template
except ImportError:
    from vllm.entrypoints.chat_utils import apply_hf_chat_template as apply_chat_template

try:
    from vllm.inputs.data import TextTokensPrompt, TokensPrompt
except ImportError:
    # API change since vLLM v0.5.3+
    from vllm.entrypoints.openai.serving_engine import TextTokensPrompt
# except ImportError:
#     print("vllm not installed")
T = TypeVar("T")
ERRORS = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )

class VLLMOfflineEngine2(LM):
    def __init__(
        self,
        llm,
        model_type: str = "chat",
        tokenizer: AutoTokenizer = None,
        batch_size: int = 2,
        **kwargs,
    ):
        super().__init__("vllm")
        self.provider = "vllm"
        self.llm = llm
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "model": llm,
            **kwargs,
        }
        self.system_prompt = None
        self.model_type = model_type
        self.history = []
        self.batch_size = 256
        self.batch_timeout = 5
        self.tokenizer = tokenizer
        async def model_config_wrapper():
            return await llm.get_model_config()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.model_config = loop.run_until_complete(model_config_wrapper())
        loop.close()


    # def _batch_request(self, prompts: List[str]) -> List[Any]:
    #     messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    #     if self.system_prompt:
    #         for msg in messages:
    #             msg.insert(0, {"role": "system", "content": self.system_prompt})
        
    #     sampling_param_dict = {k: v for k, v in self.kwargs.items() if k in msgspec.structs.asdict(SamplingParams())}
    #     sampling_params = SamplingParams(**sampling_param_dict)
    #     unique_kwargs = {k: v for k, v in self.kwargs.items() if k not in sampling_param_dict}
    #     unique_kwargs.pop("async_mode", None)
    #     unique_kwargs.pop("model", None)
    #     lora_request = LoRARequest("lora2", 1, "/home/ray/default/lora2")
    #     with self.lock_of_async_defeat:
    #         responses = chat_request(self.llm, messages, sampling_params, lora_request, **unique_kwargs)
    #     return responses

    def basic_request(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, [{"role": "system", "content": self.system_prompt}])
        
        sampling_param_dict = {k: v for k, v in self.kwargs.items() if k in msgspec.structs.asdict(SamplingParams())}
        sampling_params = SamplingParams(**sampling_param_dict)
        unique_kwargs = {k: v for k, v in self.kwargs.items() if k not in sampling_param_dict}
        unique_kwargs.pop("async_mode", None)
        # print("sending prompt)
        y = chat_request(self.llm, self.tokenizer, messages, sampling_params, lora_request=None, use_tqdm=False, **unique_kwargs, model_config=self.model_config)
        
        return y

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        
        response = self.basic_request(prompt, **kwargs)
        # outputs = response[0][0]
        choices = response.outputs

        completed_choices = [c for c in choices if c.finish_reason != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        if kwargs.get("logprobs", False):
            completions = [{'text': c.text, 'logprobs': c.logprobs} for c in choices]
        else:
            completions = [c.text for c in choices]
        # decoded_output = self.tokenizer.decode(choices[0].token_ids)
        # print(decoded_output)
        
        return completions

    def shutdown(self):
        for _ in self.batch_threads:
            self.prompt_queue.put((None, None))  # Sentinel to stop threads
        for thread in self.batch_threads:
            thread.join()
            
    @classmethod
    def instantiate_with_llm(cls, model, tokenizer=None, engine_args={}, do_download_model=True, **kwargs):
        if do_download_model:
            model_path = download_model(model)
        else:
            model_path = model

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        engine_args = AsyncEngineArgs(
            model,
            **engine_args,
            disable_log_requests=True,
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        return cls(llm, tokenizer=tokenizer, **kwargs)

# @custom_lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
# @CacheMemory.cache
def chat_request(llm: AsyncLLMEngine, tokenizer, prompt, sampling_params, lora_request, **kwargs):
    model_config =  kwargs.pop("model_config")

    prompt = parse_chat_messages(prompt, model_config, tokenizer)[0]
    templated_prompt = tokenizer.apply_chat_template(
        conversation=prompt,  # type: ignore[arg-type]
        chat_template=None,
        tokenize=False,
        **kwargs,
    )
    prompt_token_ids = tokenizer.encode(templated_prompt)
    tokens_prompt = TextTokensPrompt(prompt_token_ids=prompt_token_ids)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def wrapper():
        return await chat_request_async(llm, tokens_prompt, sampling_params, lora_request, **kwargs)
    
    try:
        return loop.run_until_complete(wrapper())
    finally:
        loop.close()


async def chat_request_async(engine: AsyncLLMEngine, prompt, sampling_params, lora_request, **kwargs):
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)
    # results_generator = iterate_with_cancellation(
    #     results_generator, is_cancelled=lambda: False)

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return None

    assert final_output is not None
    # prompt = final_output.prompt
    # assert prompt is not None
    # print(final_output)
    return final_output

# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
# @CacheMemory.cache
def completions_request(llm: AsyncLLMEngine, prompt, sampling_params, **kwargs):
    return llm.generate(prompt, sampling_params, use_tqdm=False, **kwargs)


# if __name__ == "__main__":
#     # model_path = download_model("meta-llama/Meta-Llama-3-70B-Instruct")

#     import msgspec
#     print(isinstance(SamplingParams(), msgspec.Struct))
#     print(msgspec.structs.asdict(SamplingParams()))

async def iterate_with_cancellation(
    iterator: AsyncGenerator[T, None],
    is_cancelled: Callable[[], Awaitable[bool]],
) -> AsyncGenerator[T, None]:
    """Convert async iterator into one that polls the provided function
    at least once per second to check for client cancellation.
    """

    # Can use anext() in python >= 3.10
    awaits = [ensure_future(iterator.__anext__())]
    while True:
        done, pending = await asyncio.wait(awaits, timeout=1)
        if await is_cancelled():
            with contextlib.suppress(BaseException):
                awaits[0].cancel()
                await iterator.aclose()
            raise asyncio.CancelledError("client cancelled")
        if done:
            try:
                item = await awaits[0]
                awaits[0] = ensure_future(iterator.__anext__())
                yield item
            except StopAsyncIteration:
                # we are done
                return

def random_uuid() -> str:
    return str(uuid.uuid4().hex)