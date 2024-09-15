import os
import ujson
import functools
from .base_lm import BaseLM
from typing import Literal
import requests
try:
    import vllm
except ImportError:
    vllm = None

# NOTE: Command to run the vllm server
# vllm serve Qwen/Qwen2-VL-7B-Instruct  --trust-remote-code --limit-mm-per-prompt image=8

class VLLMLM(BaseLM):
    def __init__(
        self,
        model,
        port,
        model_type: Literal["chat", "text"] = "chat",
        url="http://localhost",
        http_request_kwargs=None,
        **kwargs,
    ):
        super().__init__(model=model, is_client=True)

        if isinstance(url, list):
            self.urls = url
        elif isinstance(url, str):
            self.urls = [f"{url}:{port}"]
        else:
            raise ValueError(
                f"The url provided to `VLLMLM` is neither a string nor a list of strings. It is of type {type(url)}."
            )

        self.urls_const = tuple(self.urls)
        self.port = port
        self.http_request_kwargs = http_request_kwargs or {}
        self.model_type = model_type
        self.headers = {"Content-Type": "application/json"}
        self.kwargs = kwargs
        self.kwargs.update(
            {
                "port": port,
                "url": self.urls_const,
            }
        )

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        # Make the request and handle LRU & disk caching.
        if self.model_type == "chat":
            completion = cached_vllm_completion if cache else vllm_completion
        else:
            completion = cached_vllm_text_completion if cache else vllm_text_completion
        print(messages)
        response = completion(ujson.dumps(dict(model=self.model, messages=messages, **kwargs)))
        # Handle different response structures flexibly
        outputs = []
        for choice in response["choices"]:
            if isinstance(choice, dict):
                if "message" in choice and isinstance(choice["message"], dict):
                    content = choice["message"].get("content")
                    if isinstance(content, list):
                        outputs.append([item.get("text", "") if isinstance(item, dict) else item for item in content])
                    else:
                        outputs.append(content)
                elif "text" in choice:
                    outputs.append(choice["text"])
            elif hasattr(choice, "message"):
                content = choice.message.content
                if isinstance(content, list):
                    outputs.append([item.get("text", "") if isinstance(item, dict) else item for item in content])
                else:
                    outputs.append(content)
            else:
                outputs.append(str(choice))

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(prompt=prompt, messages=messages, kwargs=kwargs, response=response, outputs=outputs)
        entry = dict(**entry, usage=dict(response["usage"]), cost=response["_hidden_params"].get("response_cost"))
        self.history.append(entry)

        return outputs

@functools.lru_cache(maxsize=None)
def cached_vllm_completion(request):
    return vllm_completion(request, cache={"no-cache": False, "no-store": False})

def vllm_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)
    
    # Round robin the urls.
    url = kwargs['url'][0]
    kwargs['url'] = kwargs['url'][1:] + [kwargs['url'][0]]

    list_of_elements_to_allow = [
        "n", "best_of", "presence_penalty", "frequency_penalty", "repetition_penalty",
        "temperature", "top_p", "top_k", "min_p", "seed", "use_beam_search",
        "length_penalty", "early_stopping", "stop", "stop_token_ids",
        "include_stop_str_in_output", "ignore_eos", "max_tokens", "min_tokens",
        "logprobs", "prompt_logprobs", "detokenize", "skip_special_tokens",
        "spaces_between_special_tokens", "logits_processors", "truncate_prompt_tokens",
    ]
    req_kwargs = {k: v for k, v in kwargs.items() if k in list_of_elements_to_allow}

    system_prompt = kwargs.get("system_prompt", None)
    messages = kwargs.get("messages", [])
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    payload = {
        "model": kwargs["model"],
        "messages": messages,
        **req_kwargs,
    }

    response = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        **kwargs.get('http_request_kwargs', {})
    )

    try:
        json_response = response.json()
        completions = json_response["choices"]
        response = {
            "prompt": messages[-1]["content"],
            "choices": [{"message": {"content": c["message"]["content"]}} for c in completions],
            "usage": json_response.get("usage", {}),
            "_hidden_params": {"response_cost": None}
        }
        return response
    except Exception:
        print("Failed to parse JSON response:", response.text)
        raise Exception("Received invalid JSON response from server")

@functools.lru_cache(maxsize=None)
def cached_vllm_text_completion(request):
    return vllm_text_completion(request, cache=None)

def vllm_text_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)
    
    # Round robin the urls.
    url = kwargs['url'][0]
    kwargs['url'] = kwargs['url'][1:] + [kwargs['url'][0]]

    list_of_elements_to_allow = [
        "n", "best_of", "presence_penalty", "frequency_penalty", "repetition_penalty",
        "temperature", "top_p", "top_k", "min_p", "seed", "use_beam_search",
        "length_penalty", "early_stopping", "stop", "stop_token_ids",
        "include_stop_str_in_output", "ignore_eos", "max_tokens", "min_tokens",
        "logprobs", "prompt_logprobs", "detokenize", "skip_special_tokens",
        "spaces_between_special_tokens", "logits_processors", "truncate_prompt_tokens",
    ]
    req_kwargs = {k: v for k, v in kwargs.items() if k in list_of_elements_to_allow}

    # Build the prompt from the messages.
    prompt = '\n\n'.join([x['content'] for x in kwargs.pop("messages")] + ['BEGIN RESPONSE:'])

    payload = {
        "model": kwargs["model"],
        "prompt": prompt,
        **req_kwargs,
    }

    response = requests.post(
        f"{url}/v1/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        **kwargs.get('http_request_kwargs', {})
    )

    try:
        json_response = response.json()
        completions = json_response["choices"]
        response = {
            "prompt": prompt,
            "choices": [{"text": c["text"]} for c in completions],
            "usage": json_response.get("usage", {}),
            "_hidden_params": {"response_cost": None}
        }
        return response
    except Exception:
        print("Failed to parse JSON response:", response.text)
        raise Exception("Received invalid JSON response from server")