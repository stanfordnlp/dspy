import datetime
import hashlib
from typing import Any, Literal, Optional

import requests

from dsp.modules.lm import LM


def post_request_metadata(model_name, prompt):
    """Creates a serialized request object for the Ollama API."""
    timestamp = datetime.datetime.now().timestamp()
    id_string = str(timestamp) + model_name + prompt
    hashlib.sha1().update(id_string.encode("utf-8"))
    id_hash = hashlib.sha1().hexdigest()
    return {"id": f"chatcmpl-{id_hash}", "object": "chat.completion", "created": int(timestamp), "model": model_name}


class OllamaLocal(LM):
    """Wrapper around a locally hosted Ollama model (API: https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values and https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-completion).
    Returns dictionary info in the OpenAI API style (https://platform.openai.com/docs/api-reference/chat/object).

    Args:
        model (str, optional): Name of Ollama model. Defaults to "llama2".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        base_url (str):  Protocol, host name, and port to the served ollama model. Defaults to "http://localhost:11434" as in ollama docs.
        timeout_s (float): Timeout period (in seconds) for the post request to llm.
        format (str): The format to return a response in. Currently the only accepted value is `json`
        system (str): System Prompt to use when running in `text` mode.
        **kwargs: Additional arguments to pass to the API.
    """

    def __init__(
        self,
        model: str = "llama2",
        model_type: Literal["chat", "text"] = "text",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120,
        temperature: float = 0.0,
        max_tokens: int = 150,
        top_p: int = 1,
        top_k: int = 20,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        n: int = 1,
        num_ctx: int = 1024,
        format: Optional[Literal["json"]] = None,
        system: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)

        self.provider = "ollama"
        self.model_type = model_type
        self.base_url = base_url
        self.model_name = model
        self.timeout_s = timeout_s
        self.format = format
        self.system = system

        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "num_ctx": num_ctx,
            **kwargs,
        }

        # Ollama uses num_predict instead of max_tokens
        self.kwargs["num_predict"] = self.kwargs["max_tokens"]

        self.history: list[dict[str, Any]] = []
        self.version = kwargs["version"] if "version" in kwargs else ""

        # Ollama occasionally does not send `prompt_eval_count` in response body.
        # https://github.com/stanfordnlp/dspy/issues/293
        self._prev_prompt_eval_count = 0

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
        
        # Set the format if it was defined
        if self.format:
            settings_dict["format"] = self.format
            
        if self.model_type == "chat":
            settings_dict["messages"] = [{"role": "user", "content": prompt}]
        else:
            # Overwrite system prompt defined in modelfile
            if self.system:
                settings_dict["system"] = self.system
    
            settings_dict["prompt"] = prompt

        urlstr = f"{self.base_url}/api/chat" if self.model_type == "chat" else f"{self.base_url}/api/generate"
        tot_eval_tokens = 0
        for i in range(kwargs["n"]):
            response = requests.post(urlstr, json=settings_dict, timeout=self.timeout_s)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code != 200:
                # If the request was not successful, print an error message
                print(f"Error: CODE {response.status_code} - {response.text}")

            response_json = response.json()

            text = (
                response_json.get("message").get("content")
                if self.model_type == "chat"
                else response_json.get("response")
            )
            request_info["choices"].append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": "".join(text),
                    },
                    "finish_reason": "stop",
                },
            )
            tot_eval_tokens += response_json.get("eval_count")
        request_info["additional_kwargs"] = {k: v for k, v in response_json.items() if k not in ["response"]}

        request_info["usage"] = {
            "prompt_tokens": response_json.get("prompt_eval_count", self._prev_prompt_eval_count),
            "completion_tokens": tot_eval_tokens,
            "total_tokens": response_json.get("prompt_eval_count", self._prev_prompt_eval_count) + tot_eval_tokens,
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

        completions = [self._get_choice_text(c) for c in choices]

        return completions
    
    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}

        return self.__class__(
            model=self.model_name,
            model_type=self.model_type,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            **kwargs,
        )
