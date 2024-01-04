from dsp.modules.lm import LM
from typing import Any, Literal, Optional

import os, multiprocessing, datetime, hashlib
import requests

import json

def post_request_metadata(model_name, prompt):
    """Creates a serialized request object for the Ollama API."""
    timestamp = datetime.datetime.now().timestamp()
    id_string = str(timestamp) + model_name + prompt
    hashlib.sha1().update(id_string.encode("utf-8"))
    id_hash = hashlib.sha1().hexdigest()
    return {
        "id": f"chatcmpl-{id_hash}",
        "object": "chat.completion",
        "created": int(timestamp),
        "model": model_name
    }


class OllamaLocal(LM):
    """Wrapper around a locally hosted Ollama model (API: https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values and https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-completion). 
    Returns dictionary info in the OpenAI API style (https://platform.openai.com/docs/api-reference/chat/object).

    Args:
        model (str, optional): Name of Ollama model. Defaults to "llama2".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API.
    """
    
    def __init__(
        self,
        model: str = "llama2", 
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "ollama"

        self.base_url = "http://localhost:11434"  # where the model is hosted

        default_model_type = "text"
        self.model_type = model_type if model_type else default_model_type

        self.model_name = model
        self.num_cores = multiprocessing.cpu_count()-1
        self.timeout_duration = 15.0  # Seconds to wait for http requests to Ollama server

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1, 
            "top_k": 20, 
            "frequency_penalty": 0,
            "presence_penalty": 0, 
            "n": 1, 
            "num_ctx": 1024, 
            **kwargs,
        } 
        self.kwargs["num_predict"] = self.kwargs["max_tokens"]   # Ollama uses num_predict instead of max_tokens

        self.history: list[dict[str, Any]] = []
        
        self.version = ''
        if 'version' in kwargs:
            self.version = kwargs['version']

    def basic_request(self, prompt: str, **kwargs):
        
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}

        request_info = post_request_metadata(self.model_name, prompt)
        request_info["choices"] = []
        settings_dict = {
            "model": self.model_name,
            "options": {k: v for k, v in kwargs.items() if k not in ["n", "max_tokens"]},
            "stream": False, 
        }
        if self.model_type == "chat":
            settings_dict["messages"] = [{"role": "user", "content": prompt}]
        else:
            settings_dict["prompt"] = prompt
        
        urlstr = f"{self.base_url}/api/chat" if self.model_type == "chat" else f"{self.base_url}/api/generate"
        tot_eval_tokens = 0
        for i in range(kwargs["n"]):
            response = requests.post(urlstr, json=settings_dict)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code != 200:
                # If the request was not successful, print an error message
                print(f"Error: CODE {response.status_code} - {response.text}")
            
            response_json = response.json()
            
            text = response_json.get("message").get("content") if self.model_type == "chat" else response_json.get("response")
            request_info["choices"].append({
                "index": i,
                "message": { "role": "assistant", "content": ''.join(text), }, 
                "finish_reason": "stop"
            })
            tot_eval_tokens += response_json.get("eval_count")
        request_info["additional_kwargs"] = {k: v for k, v in response_json.items() if k not in ["response"]}

        print('RESPONSE JSON', response_json)
        request_info["usage"] = {
            "prompt_tokens": response_json.get("prompt_eval_count"),
            "completion_tokens": tot_eval_tokens,
            "total_tokens": response_json.get("prompt_eval_count") + tot_eval_tokens
        }

        history = {
            "prompt": prompt,
            "response": request_info,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs, 
        }
        self.history.append(history)

        return request_info

    def request(self, prompt: str, **kwargs):
        """Wrapper for requesting completions from the Ollama model."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        return choice["message"]["content"]
    
    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from Ollama.

        Args:
            prompt (str): prompt to send to Ollama
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.request(prompt, **kwargs)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]
            
        if only_completed and len(completed_choices):
            choices = completed_choices
        print(choices)
        completions = [self._get_choice_text(c) for c in choices]

        return completions