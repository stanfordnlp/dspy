import os
from typing import Any, Literal, Optional

import requests

from dsp.modules.lm import LM

SMART_ENDPOINT = "https://chat-api.you.com/smart"
RESEARCH_ENDPOINT = "https://chat-api.you.com/research"


class You(LM):
    """Wrapper around You.com's conversational Smart and Research APIs.

    Each API endpoint is designed to generate conversational
    responses to a variety of query types, including inline citations
    and web results when relevant.

    Smart Mode:
    - Quick, reliable answers for a variety of questions
    - Cites the entire web page URL

    Research Mode:
    - In-depth answers with extensive citations for a variety of questions
    - Cites the specific web page snippet relevant to the claim

    To connect to the You.com api requires an API key which
    you can get at https://api.you.com.

    For more information, check out the documentations at
    https://documentation.you.com/api-reference/.

    Args:
        endpoint: You.com conversational endpoints. Choose from "smart" or "research"
        api_key: You.com API key, if `YDC_API_KEY` is not set in the environment
    """

    def __init__(
        self,
        endpoint: Literal["smart", "research"] = "smart",
        ydc_api_key: Optional[str] = None,
    ):
        super().__init__(model="you.com")
        self.ydc_api_key = ydc_api_key or os.environ["YDC_API_KEY"]
        self.endpoint = endpoint

        # Mandatory DSPy attributes to inspect LLM call history
        self.history = []
        self.provider = "you.com"

    def basic_request(self, prompt, **kwargs) -> dict[str, Any]:
        headers = {"x-api-key": self.ydc_api_key}
        params = {"query": prompt}  # DSPy `kwargs` are ignored as they are not supported by the API

        response = requests.post(self.request_endpoint, headers=headers, json=params)
        response.raise_for_status()

        data = response.json()

        # Update history
        self.history.append({"prompt": prompt, "response": data, "endpoint": self.endpoint})

        return data

    @property
    def request_endpoint(self) -> str:
        if self.endpoint == "smart":
            return SMART_ENDPOINT
        return RESEARCH_ENDPOINT

    def __call__(self, prompt, only_completed: bool = True, return_sorted: bool = False, **kwargs) -> list[str]:
        response = self.request(prompt, **kwargs)
        return [response["answer"]]
