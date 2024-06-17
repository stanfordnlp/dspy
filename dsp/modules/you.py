import os
from typing import Any, Literal, Optional

import requests

from dsp.modules.lm import LM

SMART_ENDPOINT = "https://chat-api.you.com/smart"
RESEARCH_ENDPOINT = "https://chat-api.you.com/research"


class You(LM):

    def __init__(
        self,
        mode: Literal["smart", "research"] = "smart",
        api_key: Optional[str] = None,
    ):
        super().__init__(model="you.com")
        self.api_key = api_key or os.environ["YDC_API_KEY"]
        self.mode = mode

        # Mandatory DSPy attributes to inspect LLM call history
        self.history = []
        self.provider = "you.com"

    def basic_request(self, prompt, **kwargs) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key}
        params = {"query": prompt}  # DSPy `kwargs` are ignored as they are not supported by the API

        response = requests.post(self.endpoint, headers=headers, json=params)
        response.raise_for_status()

        data = response.json()

        # Update history
        self.history.append({"prompt": prompt, "response": data, "mode": self.mode})

        return data

    @property
    def endpoint(self) -> str:
        if self.mode == "smart":
            return SMART_ENDPOINT
        return RESEARCH_ENDPOINT

    def __call__(self, prompt, only_completed: bool = True, return_sorted: bool = False, **kwargs) -> list[str]:
        response = self.request(prompt, **kwargs)
        return [response["answer"]]
