import dspy
import os
import requests
from dsp.utils import dotdict

from typing import Union, List, Optional


class YouRM(dspy.Retrieve):
    def __init__(self, ydc_api_key=None, k=3):
        super().__init__(k=k)
        if not ydc_api_key and not os.environ.get("YDC_API_KEY"):
            raise RuntimeError("You must supply ydc_api_key or set environment variable YDC_API_KEY")
        elif ydc_api_key:
            self.ydc_api_key = ydc_api_key
        else:
            self.ydc_api_key = os.environ["YDC_API_KEY"]

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        """Search with You.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of context strings to return, if not already specified in self.k

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """

        k = k if k is not None else self.k

        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        docs = []
        for query in queries:
            headers = {"X-API-Key": self.ydc_api_key}
            results = requests.get(
                f"https://api.ydc-index.io/search?query={query}",
                headers=headers,
            ).json()
            for hit in results["hits"][:k]:
                for snippet in hit["snippets"]:
                    docs.append(snippet)
        return [dotdict({"long_text": document}) for document in docs]
