from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs, RequestOutput
from vllm.entrypoints.chat_utils import parse_chat_messages, apply_chat_template

from typing import List, Dict, Optional, Union

import asyncio
try:
    from vllm.inputs.data import TextTokensPrompt
except ImportError:
    # API change since vLLM v0.5.3+
    from vllm.entrypoints.openai.serving_engine import TextTokensPrompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import dotenv
dotenv.load_dotenv()

class AsyncLLMWrapper:
    def __init__(self, *args, max_pending_requests: int = -1, tokenizer: AutoTokenizer = None, **kwargs):
        self.loop = asyncio.get_event_loop()

        kwargs["tokenizer"] = kwargs.get("tokenizer", "meta-llama/Meta-Llama-3-70B-Instruct")
        engine_args = AsyncEngineArgs(
            *args,
            **kwargs,
            disable_log_requests=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.request_id = 0
        self.max_pending_requests = max_pending_requests
        self.tokenizer = tokenizer
        self.lock = asyncio.Lock()

        # vLLM performance gets really bad if there are too many requests in the pending queue.
        # We work around it by introducing another queue that gates how many requests we are
        # sending to vLLM at once.
        # This is not a queue of requests - instead, this queue holds "slots". Each time
        # we add a new request, we take one slot. Each time a request finishes, we add a new
        # slot.
        self.free_queue = asyncio.Queue()
        if self.max_pending_requests > 0:
            for _ in range(self.max_pending_requests):
                self.free_queue.put_nowait(True)

    async def _process(
        self, request_id, prompt, prompt_token_ids, sampling_params, idx_in_batch
    ):
        if isinstance(prompt, str):
            assert prompt_token_ids
            prompt = TextTokensPrompt(prompt=prompt, prompt_token_ids=prompt_token_ids)
        if self.max_pending_requests > 0:
            await self.free_queue.get()
        print("adding request with id", request_id)
        stream = await self.engine.add_request(str(request_id), prompt, sampling_params)
        async for request_output in stream:
            if request_output.finished:
                if self.max_pending_requests > 0:
                    self.free_queue.put_nowait(True)
                return (request_output, idx_in_batch)
            else:
                print("request_output", request_output)
        assert False

    async def start_generate_async(self, prompt, sampling_params, prompt_token_ids):
        tasks = []

        for i, p in enumerate(prompt):
            self.request_id += 1
            if prompt_token_ids:
                pti = prompt_token_ids[i]
                p = TextTokensPrompt(prompt=p, prompt_token_ids=pti)
            else:
                assert p["prompt_token_ids"]
                pti = None
            tasks.append(
                    self._process(self.request_id, p, pti, sampling_params, i)
            )
            # async for request_output in stream:
            # if request_output.finished:
            #     if self.max_pending_requests > 0:
            #         self.free_queue.put_nowait(True)
            #     return (request_output, idx_in_batch)
            
        return tasks

    async def generate_async(self, prompt, sampling_params, prompt_token_ids):
        tasks = await self.start_generate_async(
            prompt, sampling_params, prompt_token_ids
        )
        returns = await asyncio.gather(*tasks)
        return returns

    async def chat_async(
        self,
        messages: List[List[Dict[str, str]]],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request = None, #TODO
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[RequestOutput]:
        # async with self.lock:
        model_config = await asyncio.wait_for(self.engine.get_model_config(), timeout=10)
        conversations= [parse_chat_messages(conversation, model_config,
                                               self.tokenizer)[0] for conversation in messages]
        prompts = [apply_chat_template(
            self.tokenizer,
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt) for conversation in conversations]

        prompt_token_ids = [self.tokenizer.encode(prompt) for prompt in prompts]

        tasks = self.start_generate_async(
            prompts, sampling_params, prompt_token_ids
        )
        
        returns = await asyncio.gather(*tasks, return_exceptions=True)
        return returns

    def chat(self, messages: List[List[Dict[str, str]]], sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None, use_tqdm: bool = True, lora_request = None, chat_template: Optional[str] = None, add_generation_prompt: bool = True):
        ret = self.loop.run_until_complete(
            self.chat_async(messages, sampling_params, use_tqdm, lora_request, chat_template, add_generation_prompt)
        )
        return ret

    async def chat_single(self, messages: List[Dict[str, str]], sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None, use_tqdm: bool = True, lora_request = None, chat_template: Optional[str] = None, add_generation_prompt: bool = True):
        async with self.lock:
            self.request_id += 1
            request_id = self.request_id
            
        model_config = await asyncio.wait_for(self.engine.get_model_config(), timeout=10)
        print("messages", messages)
        conversation= parse_chat_messages(messages, model_config,
                                               self.tokenizer)[0]
        prompt = apply_chat_template(
            self.tokenizer,
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt)

        prompt_token_ids = self.tokenizer.encode(prompt)

        x = await self._process(request_id, prompt, prompt_token_ids, sampling_params, 0)
        return x

if __name__ == "__main__":
    engine = AsyncLLMWrapper(model="meta-llama/Meta-Llama-3-8B-Instruct", enforce_eager=True, engine_use_ray=False, worker_use_ray=False)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    # engine.generate("What is LLM?", SamplingParams(temperature=0.0), None)
    prompt = "What is LLM?"
    sampling_params = SamplingParams(temperature=0.0)
    prompt_token_ids = tokenizer.encode(prompt)
    print(next(engine.generate([prompt], sampling_params, [prompt_token_ids])))