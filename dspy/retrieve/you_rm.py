import os
import warnings
from typing import Any, Literal, Optional, Union

import requests

import dspy
from dsp.utils import dotdict


class YouRM(dspy.Retrieve):
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
        """Search with You.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, list[str]]): The query or queries to search for.
            k (Optional[int]): The number of context strings to return, if not already specified in self.k  # TODO: Clarify

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """

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
