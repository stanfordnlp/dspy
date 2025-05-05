import functools
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import requests
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import dspy
from dsp.modules.cache_utils import CacheMemory, cache_turn_on
from dspy.primitives.prediction import Prediction


class BingSearch(dspy.Retrieve):
    EMBEDDING_MODEL = "avsolatorio/GIST-small-Embedding-v0"
    MAX_EMB_SEQ_LEN = 512
    DEFAULT_SEARCH_COUNT = 10

    def __init__(
        self, api_key: Optional[str] = None, endpoint: Optional[str] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY")
            if api_key is None:
                raise ValueError("BING_SEARCH_V7_SUBSCRIPTION_KEY is not set")
        if endpoint is None:
            endpoint = os.environ.get("BING_SEARCH_V7_ENDPOINT")
            if endpoint is None:
                raise ValueError("BING_SEARCH_V7_ENDPOINT is not set")

        self.api_key = api_key
        self.endpoint = endpoint
        self.cache_path = "bing_cache.db"
        self.model = SentenceTransformer(self.EMBEDDING_MODEL)

    def forward(self, query: str, count: int = 10) -> Optional[Dict[str, Any]]:
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": self.DEFAULT_SEARCH_COUNT, "mkt": "en-us"}

        endpoint = self.endpoint + "/v7.0/search"

        try:
            response = requests.get(endpoint, headers=headers, params=params)
        except Exception as e:
            print(e)
            raise e

        response_json = response.json()
        if "webPages" not in response_json:
            # no search results came back
            raise ValueError("No search results came back")
        num_results = len(response_json["webPages"]["value"])
        snippets = [
            response_json["webPages"]["value"][i]["snippet"]
            for i in range(num_results)
        ]

        word_counts = self._get_word_count(query, snippets)
        similarities = self._get_similarity_scores(query, snippets)
        # Trim the response to only include the first count results
        # since we default to searching for DEFAULT_SEARCH_COUNT
        trimmed_response = response_json
        trimmed_response["webPages"]["value"] = response_json["webPages"]["value"][
            0:count
        ]
        passages = [
            trimmed_response["webPages"]["value"][i]["snippet"]
            for i in range(len(trimmed_response["webPages"]["value"]))
        ]
        return dspy.Prediction(passages=passages)

    def entity(self, query: str) -> dict:

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "mkt": "en-us"}

        endpoint = self.endpoint + "/v7.0/entities"

        try:
            response = requests.get(endpoint, headers=headers, params=params)
            return response.json()
        except Exception as e:
            print(e)
            raise e

    def _get_similarity_scores(self, query: str, snippets: List[str]) -> List[float]:
        n = len(snippets)
        queries = [query] * n
        snips = snippets.copy()

        query_embeddings = self.model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False,
        )
        snip_embeddings = self.model.encode(
            snips, convert_to_tensor=True, show_progress_bar=False,
        )

        cos_sims = F.cosine_similarity(query_embeddings, snip_embeddings, dim=1)

        return cos_sims.tolist()

    def _split_str_and_remove_punctuation(self, string: str) -> List[str]:
        string = string.split(" ")
        string = list(map(lambda word: word if word.isalpha() else word[:-1], string))
        return string

    def _get_word_count(self, query: str, snippets: List[str]) -> List[int]:
        query_words = self._split_str_and_remove_punctuation(query)
        query_words = [query_words] * len(snippets)
        snippets_words = list(map(self._split_str_and_remove_punctuation, snippets))

        pairs = list(zip(query_words, snippets_words))

        word_counts = []

        for q_words, s_words in pairs:
            q = Counter(q_words)
            s = Counter(s_words)

            matching_count = 0
            for word, count in q.items():
                if word in s:
                    words_used = min(count, s[word])
                    matching_count += words_used

            word_counts.append(matching_count)

        return word_counts


class BraveSearch(dspy.Retrieve):
    """Set API key in BRAVE_SEARCH_API_KEY
    
    Return result: Prediction(list[dict["title", "link", "snippet"]]) 
    """
    api_key: str
    base_url: str = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key=None) -> None:
        if api_key is None:
            self.api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
            if self.api_key is None:
                raise ValueError("BRAVE_SEARCH_API_KEY is not set")

    def forward(self, query: str, count=10) -> Prediction:
        web_search_results = self._search_request(query=query, count=count)
        final_results = [
            {
                "title": item.get("title"),
                "link": item.get("url"),
                "snippet": item.get("description"),
            }
            for item in web_search_results
        ]
        return Prediction(passages=final_results)


    # Credit to LangChain
    def _search_request(self, query: str, **kwargs) -> List[dict]:
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        req = requests.PreparedRequest()
        params = {**kwargs, **{"q": query}}
        req.prepare_url(self.base_url, params)
        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")
        

        # Retry 3 times
        for _ in range(3):
            response = cached_brave_search_request_wrapped(req.url, **headers)
            if not response.ok:
                dspy.logger.error(f"HTTP error: {response.status_code}, {response.text}")
                headers["Cache-Control"] = "no-cache"
                dspy.logger.error(f"req.url: {req.url}")
            else:
                break
        
        try:
            return response.json().get("web", {}).get("results", [])
        except Exception as e:
            dspy.logger.error(f"Error in parsing response: {e}")
            return []


@CacheMemory.cache
def cached_brave_search_request(url, **header):
    response = requests.get(url, headers=header)
    return response


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
def cached_brave_search_request_wrapped(url, **header):
    return cached_brave_search_request(url, **header)