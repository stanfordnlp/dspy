import os
from typing import Any, Literal, Optional

import requests

from dsp.modules.lm import LM

SMART_ENDPOINT = ""  # TODO: Update before submitting PR
RESEARCH_ENDPOINT = ""  # TODO: Update before submitting PR


class YouLM(LM):  # TODO: Should this be `You` instead?
    """# TODO[DOCME]"""

    def __init__(
        self,
        mode: Literal["smart", "research"] = "smart",
        api_key: Optional[str] = None,
    ):
        super().__init__(model="you.com")

        if not api_key:
            api_key = os.environ["YDC_API_KEY"]

        self.api_key = api_key
        self.mode = mode

        # Mandatory DSPy attributes to inspect LLM call history
        self.history = []
        self.provider = "you.com"

    def basic_request(self, prompt, **kwargs) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key}
        params = {"query": prompt}  # DSPy `kwargs` are ignored as they are not supported by the API

        response = requests.get(self.endpoint, headers=headers, params=params)
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

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs) -> list[str]:
        response = self.request(prompt, **kwargs)
        return [response["answer"]]
