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
from deploy_dspy.download import download_model
from deploy_dspy.async_llm import AsyncLLMWrapper
from transformers import AutoTokenizer
try:
    import msgspec
    from vllm import LLM, SamplingParams
    # from vllm.lora.request import LoraRequest

except ImportError:
    print("vllm not installed")

ERRORS = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )

class VLLMOfflineEngine(LM):
    def __init__(
        self,
        llm,
        model_type: str = "chat",
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
            **kwargs,
        }
        self.system_prompt = None
        self.model_type = model_type
        self.history = []
        self.batch_size = 256
        self.batch_timeout = 10
        self.max_concurrent_batches = 2
        self.prompt_queue = queue.Queue()
        self.result_dict = {}
        self.result_lock = Lock()
        self.batch_result_counts = [0 for _ in range(self.max_concurrent_batches)]
        self.batch_locks = [Lock() for _ in range(self.max_concurrent_batches)]
        self.batch_events = [Event() for _ in range(self.max_concurrent_batches)]
        self.batch_threads = [Thread(target=self._process_batch, args=(i,)) for i in range(self.max_concurrent_batches)]
        
        for thread in self.batch_threads:
            thread.start()


    def _process_batch(self, batch_id):
        while True:
            prompts = []
            thread_ids = []
            start_time = time.time()
            
            while len(prompts) < self.batch_size and (time.time() - start_time) < self.batch_timeout:
                try:
                    prompt, thread_id = self.prompt_queue.get(timeout=self.batch_timeout - (time.time() - start_time))
                    prompts.append(prompt)
                    thread_ids.append(thread_id)
                except queue.Empty:
                    break
            
            if prompts:
                with self.batch_locks[batch_id]:
                    responses = self._batch_request(prompts)
                    
                    with self.result_lock:
                        for thread_id, response in zip(thread_ids, responses):
                            self.result_dict[thread_id] = response
                        self.batch_result_counts[batch_id] = len(responses)
                
                self.batch_events[batch_id].set()


    def _batch_request(self, prompts: List[str]) -> List[Any]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        if self.system_prompt:
            for msg in messages:
                msg.insert(0, {"role": "system", "content": self.system_prompt})
        
        sampling_param_dict = {k: v for k, v in self.kwargs.items() if k in msgspec.structs.asdict(SamplingParams())}
        sampling_params = SamplingParams(**sampling_param_dict)
        unique_kwargs = {k: v for k, v in self.kwargs.items() if k not in sampling_param_dict}
        unique_kwargs.pop("async_mode", None)
        responses = self.llm.chat(messages, sampling_params, use_tqdm=False, **unique_kwargs)
        return responses

    def basic_request(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, [{"role": "system", "content": self.system_prompt}])
        sampling_param_dict = {k: v for k, v in self.kwargs.items() if k in msgspec.structs.asdict(SamplingParams())}
        sampling_params = SamplingParams(**sampling_param_dict)
        unique_kwargs = {k: v for k, v in self.kwargs.items() if k not in sampling_param_dict}
        unique_kwargs.pop("async_mode", None)
        print("sending prompt: ", messages)
        y = asyncio.run(self.llm.chat_single(messages, sampling_params, use_tqdm=False, **unique_kwargs))
        print("y", y)
        return y
        # thread_id = id(prompt)
        # self.prompt_queue.put((prompt, thread_id))
        
        # while True:
        #     for batch_id, event in enumerate(self.batch_events):
        #         if event.is_set():
        #             with self.result_lock:
        #                 if thread_id in self.result_dict:
        #                     response = self.result_dict.pop(thread_id)
        #                     self.batch_result_counts[batch_id] -= 1
        #                     if self.batch_result_counts[batch_id] == 0:
        #                         print(f"Batch {batch_id} completed")
        #                         event.clear()
        #                     return response
        #     time.sleep(0.1)

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
        llm = AsyncLLMWrapper(model_path,
                max_pending_requests=engine_args.pop("max_pending_requests", 512),
                tokenizer=tokenizer,
                **engine_args
            )
        return cls(llm, **kwargs)

def chat_request(llm: LLM, prompt, sampling_params, **kwargs):
    x = llm.chat(prompt, sampling_params, use_tqdm=False, **kwargs)
    return x

# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
# @CacheMemory.cache
def completions_request(llm: LLM, prompt, sampling_params, **kwargs):
    return llm.generate(prompt, sampling_params, use_tqdm=False, **kwargs)