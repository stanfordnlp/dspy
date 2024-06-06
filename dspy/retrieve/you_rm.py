import os
import warnings
from typing import Any, Literal, Optional, Union

import requests

import dspy
from dsp.utils import dotdict


class YouRM(dspy.Retrieve):
    """Retriever for You.com's Search and News API.

    [API reference](https://documentation.you.com/api-reference/)

    Args:
        ydc_api_key: you.com API key, if `YDC_API_KEY` is not set in the environment
        k: If ``endpoint_type="search"``, the max snippets to return per search hit.
           If ``endpoint_type="news"``, the max articles to return.
        endpoint_type: you.com endpoints
        num_web_results: The max number of web results to return, must be under 20
        safesearch: Safesearch settings, one of "off", "moderate", "strict", defaults to moderate
        country: Country code, ex: 'US' for United States, see API reference for more info
        search_lang: (News API) Language codes, ex: 'en' for English, see API reference for more info
        ui_lang: (News API) User interface language for the response, ex: 'en' for English, see API reference for more info
        spellcheck: (News API) Whether to spell check query or not, defaults to True
    """

    def __init__(
        self,
        ydc_api_key: Optional[str] = None,
        k: int = 3,
        endpoint_type: Literal["search", "news"] = "search",
        num_web_results: Optional[int] = None,
        safesearch: Optional[Literal["off", "moderate", "strict"]] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        spellcheck: Optional[bool] = None,
    ):
        super().__init__(k=k)

        # Data validation
        if not ydc_api_key and not os.environ.get("YDC_API_KEY"):
            raise RuntimeError('You must supply `ydc_api_key` or set environment variable "YDC_API_KEY"')

        if endpoint_type not in ("search", "news"):
            raise ValueError('`endpoint_type` must be either "search" or "news"')

        # Raise warning if News API-specific fields are set but endpoint_type is not "news"
        if endpoint_type != "news":
            news_api_fields = (search_lang, ui_lang, spellcheck)
            for field in news_api_fields:
                if field:
                    warnings.warn(
                        (
                            f"News API-specific field '{field}' is set but "
                            f'`endpoint_type="{endpoint_type}"`. '
                            "This will have no effect."
                        ),
                        UserWarning,
                    )

        self.ydc_api_key = ydc_api_key or os.environ.get("YDC_API_KEY")
        self.endpoint_type = endpoint_type
        self.num_web_results = num_web_results
        self.safesearch = safesearch
        self.country = country
        self.search_lang = search_lang
        self.ui_lang = ui_lang
        self.spellcheck = spellcheck

    def _generate_params(self, query: str) -> dict[str, Any]:
        params = {"safesearch": self.safesearch, "country": self.country}

        if self.endpoint_type == "search":
            params.update(
                query=query,
                num_web_results=self.num_web_results,
            )
        elif self.endpoint_type == "news":
            params.update(
                q=query,
                count=self.num_web_results,
                search_lang=self.search_lang,
                ui_lang=self.ui_lang,
                spellcheck=self.spellcheck,
            )

        # Remove `None` values
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None) -> dspy.Prediction:
        k = k if k is not None else self.k

        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        docs: list[str]
        for query in queries:
            headers = {"X-API-Key": self.ydc_api_key}
            params = self._generate_params(query)
            response = requests.get(
                f"https://api.ydc-index.io/{self.endpoint_type}",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            results = response.json()

            if self.endpoint_type == "search":
                docs = [snippet for hits in results["hits"][:k] for snippet in hits["snippets"]]
            elif self.endpoint_type == "news":
                docs = [article["description"] for article in results["news"]["results"][:k]]
        return [dotdict({"long_text": document}) for document in docs]
