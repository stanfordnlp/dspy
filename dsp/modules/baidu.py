import logging
import os
from typing import Any, Optional
from dsp.modules.lm import LM

import qianfan

logger = logging.getLogger(__name__)


class Ernie(LM):

    def __init__(
            self,
            model: str = "ERNIE-4.0-8K",
            api_key: Optional[str] = None,
            secret_key: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(model)
        self.provider = "baidu"
        self.api_key = api_key = os.environ.get("QIANFAN_AK") if api_key is None else api_key
        self.secret_key = secret_key = os.environ.get("QIANFAN_SK") if secret_key is None else secret_key

        qianfan.AK(api_key)
        qianfan.SK(secret_key)

        self.kwargs = {
            "temperature": 0.1 if "temperature" not in kwargs else kwargs["temperature"],
            "max_output_tokens": min(kwargs.get("max_output_tokens", 2048), 2048),
            "top_p": 1.0 if "top_p" not in kwargs else kwargs["top_p"],
            "n": kwargs.pop("n", kwargs.pop("num_generations", 1)),
            **kwargs,
        }
        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []
        self.client = qianfan.ChatCompletion()

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        kwargs.pop("n")
        print(kwargs)
        response = self.client.do(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def request(self, prompt: str, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def __call__(self, prompt, **kwargs):
        n = kwargs.pop("n", 1)
        completions = []
        for i in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response.body['result'])
        return completions
