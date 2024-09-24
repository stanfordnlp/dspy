import queue
import threading
from threading import Lock, Event, Thread
from dsp.modules.lm import LM
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.utils.settings import settings
import backoff
from typing import Any, Generator, List
import functools
import asyncio
import collections
import time
import ujson
from dsp.deploy_dspy.download import download_model
from dsp.deploy_dspy.async_llm import AsyncLLMWrapper
from transformers import AutoTokenizer
# try:
import msgspec
from vllm import LLM, SamplingParams # maybe need vllm 0.5.4
from vllm.lora.request import LoRARequest

# except ImportError:
#     print("vllm not installed")

ERRORS = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )

class VLLMOfflineEngine3(LM):
    def __init__(self, llm, model_type: str = "chat", batch_size: int = 32, **kwargs):
        super().__init__("vllm")
        self.llm = llm
        self.provider = "vllm"
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
        self.batch_size = batch_size
        self.batch_timeout = 10  # Shorter timeout for more responsive batching
        self.prompt_queue = queue.Queue()
        self.result_dict = {}
        self.queue_lock = Lock()
        self.result_set_lock = Lock()
        self.result_get_lock = Lock()
        self.lock_of_async_defeat = Lock()
        self.collector_lock = Lock()
        
        self.num_batch_threads = 2
        self.batch_threads = [
            threading.Thread(target=self._process_batches, args=(i,)) for i in range(self.num_batch_threads)
        ]
        for thread in self.batch_threads:
            thread.start()

    def _process_batches(self, batch_id):
        while True:
            batch = []
            print(f"Batch {batch_id} is waiting")
            with self.collector_lock:
                print(f"Batch {batch_id} is collecting")
                start_time = time.time()
                # Collect prompts for the batch
                while len(batch) < self.batch_size / self.num_batch_threads and (time.time() - start_time) < self.batch_timeout:
                    try:
                        prompt, thread_id = self.prompt_queue.get(timeout=self.batch_timeout - (time.time() - start_time))
                        if prompt is None:  # Shutdown signal
                            return
                        batch.append((prompt, thread_id))
                    except queue.Empty:
                        if batch:  # Process the batch if we have any prompts
                            pass
                            # break
                        time.sleep(0.01)  # Short sleep if queue is empty
            
            if batch:
                print(f"Batch {batch_id} is processing {len(batch)} prompts")
                prompts, thread_ids = zip(*batch)
                responses = self._batch_request(prompts)
                print(f"Batch {batch_id} is done processing")
                with self.result_set_lock:
                    for thread_id, response in zip(thread_ids, responses):
                        self.result_dict[thread_id] = response

    def _batch_request(self, prompts: List[str]) -> List[Any]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        if self.system_prompt:
            for msg in messages:
                msg.insert(0, {"role": "system", "content": self.system_prompt})
        
        sampling_param_dict = {k: v for k, v in self.kwargs.items() if k in msgspec.structs.asdict(SamplingParams())}
        sampling_params = SamplingParams(**sampling_param_dict)
        unique_kwargs = {k: v for k, v in self.kwargs.items() if k not in sampling_param_dict}
        unique_kwargs.pop("async_mode", None)
        unique_kwargs.pop("model", None)
        # lora_request = LoRARequest("lora2", 1, "/home/ray/default/lora2")
        # with self.lock_of_async_defeat:
        lora_request = None
        responses = chat_request(self.llm, messages, sampling_params, lora_request, **unique_kwargs)
        return responses

    def basic_request(self, prompt, **kwargs):
        thread_id = id(prompt)
        
        with self.queue_lock:
            self.prompt_queue.put((prompt, thread_id))
        
        while True:
            with self.result_get_lock:
                if thread_id in self.result_dict:
                    return self.result_dict.pop(thread_id)
            time.sleep(0.01)  # Short sleep to prevent busy-waiting

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        
        response, _ = self.basic_request(prompt, **kwargs)
        # outputs = response[0][0]
        choices = response.outputs

        completed_choices = [c for c in choices if c.finish_reason != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        if kwargs.get("logprobs", False):
            completions = [{'text': c.text, 'logprobs': c.logprobs} for c in choices]
        else:
            completions = [c.text for c in choices]
        
        return completions

    def shutdown(self):
        for _ in self.batch_threads:
            self.prompt_queue.put((None, None))  # Shutdown signal
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
        llm = AsyncLLMWrapper(model_path,
                max_pending_requests=engine_args.pop("max_pending_requests", 512),
                tokenizer=tokenizer,
                **engine_args
            )
        return cls(llm, **kwargs)

def custom_lru_cache(maxsize=None):
    def decorator(func):
        @functools.lru_cache(maxsize=maxsize)
        def cached_single_prompt(llm_name: str, prompt: str, sampling_params, **kwargs):
            # Create a simple LLM-like object
            mock_llm = type('LLM', (), {'name': llm_name})()
            
            # Call the original function with a single-prompt list
            result = func(mock_llm, [prompt], sampling_params, **kwargs)
            
            # Return the first (and only) result
            return result[0]

        @functools.wraps(func)
        def wrapper(llm, prompts: List[str], sampling_params, **kwargs):
            llm_name = llm.name
            results = []
            uncached_prompts = []
            uncached_indices = []
            
            sampling_params_dict = msgspec.structs.asdict(sampling_params)
            for k, v in sampling_params_dict.items():
                if isinstance(v, set):
                    sampling_params_dict[k] = list(v)
            sampling_params_str = ujson.dumps(sampling_params_dict)

            # Check cache for each prompt
            
            for i, prompt in enumerate(prompts):
                stringified_prompt = ujson.dumps(prompt)
                # try:
                #     results.append(cached_single_prompt(llm_name, stringified_prompt, sampling_params_str, **kwargs))
                # except KeyError:
                #     uncached_prompts.append(prompt)
                #     uncached_indices.append(i)
                if (llm_name, prompt, sampling_params_str, kwargs) in cached_single_prompt.cache_info().cache:
                    results.append(cached_single_prompt(llm_name, prompt, sampling_params_str, **kwargs))
                else:
                    uncached_prompts.append(prompt)
                    uncached_indices.append(i)
            
            # Process uncached prompts
            if uncached_prompts:
                uncached_results = func(llm, uncached_prompts, sampling_params, **kwargs)
                
                # Cache new results
                for prompt, result in zip(uncached_prompts, uncached_results):
                    stringified_prompt = ujson.dumps(prompt)
                    cached_single_prompt(llm_name, stringified_prompt, sampling_params_str, **kwargs)
                
                # Merge cached and new results
                for i, result in zip(uncached_indices, uncached_results):
                    results.insert(i, result)
            
            return results
        return wrapper
    return decorator

# @custom_lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
# @CacheMemory.cache
def chat_request(llm: AsyncLLMWrapper, prompt, sampling_params, lora_request, **kwargs):
    x = llm.chat(prompt, sampling_params, use_tqdm=False, lora_request=lora_request, **kwargs)
    return x

# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
# @CacheMemory.cache
def completions_request(llm: LLM, prompt, sampling_params, **kwargs):
    return llm.generate(prompt, sampling_params, use_tqdm=False, **kwargs)


# if __name__ == "__main__":
#     # model_path = download_model("meta-llama/Meta-Llama-3-70B-Instruct")

#     import msgspec
#     print(isinstance(SamplingParams(), msgspec.Struct))
#     print(msgspec.structs.asdict(SamplingParams()))