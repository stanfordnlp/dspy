import functools
import os
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Union

import requests
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import dspy
from dsp.modules.cache_utils import CacheMemory, cache_turn_on
from dspy.primitives.prediction import Prediction


class BingSearch(dspy.Retrieve):

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        search_market: str = "en-US",
        embedding_model: str = "avsolatorio/GIST-small-Embedding-v0",
        max_emb_seq_len: int = 512,
        default_search_count: int = 10,
        sort_by: Union[Literal["default", "word_count", "similarity"], str] = "default",
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
        self.max_emb_seq_len = max_emb_seq_len
        self.default_search_count = default_search_count
        self.search_market = search_market
        if sort_by == "similarity":
            self.model = SentenceTransformer(
                embedding_model, device=self._get_default_device()
            )

    def _get_default_device(self):
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def forward(
        self, query: str, count: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        if count is None:
            count = self.default_search_count

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": self.default_search_count,
            "mkt": self.search_market,
        }

        endpoint = self.endpoint + "/v7.0/search"

        try:
            response = cached_bing_search_request_wrapped(endpoint, headers, params)
            response_json = response.json()
        except Exception as e:
            dspy.logger.error(f"Error in getting Bing search results: {e}")
            raise e

        if "webPages" not in response_json:
            dspy.logger.info("No search results came back")
            raise ValueError("No search results came back")

        snippets = [
            search_result["snippet"]
            for search_result in response_json["webPages"]["value"]
        ]

        # Trim the response to only include the first count results
        # since we default to searching for DEFAULT_SEARCH_COUNT
        trimmed_response = response_json
        trimmed_response["webPages"]["value"] = response_json["webPages"]["value"][
            0:count
        ]

        if self.sort_by == "word_count":
            word_counts = self._get_word_count(query, snippets)
            sorted_indices = sorted(
                range(len(word_counts)), key=word_counts.__getitem__
            )
        elif self.sort_by == "similarity":
            similarity_scores = self._get_similarity_scores(query, snippets)
            sorted_indices = sorted(
                range(len(similarity_scores)), key=similarity_scores.__getitem__
            )
        else:
            sorted_indices = range(count)

        passages = [
            {
                "title": trimmed_response["webPages"]["value"][i]["name"],
                "url": trimmed_response["webPages"]["value"][i]["url"],
                "snippet": trimmed_response["webPages"]["value"][i]["snippet"],
            }
            for i in sorted_indices
        ]

        return dspy.Prediction(passages=passages)

    def entity(self, query: str) -> Dict[str, Any]:

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "mkt": "en-us"}

        endpoint = self.endpoint + "/v7.0/entities"

        try:
            response = cached_bing_search_request_wrapped(endpoint, headers, params)
            return response.json()
        except Exception as e:
            dspy.logger.error(f"Error in getting Bing entity results: {e}")
            raise e

    def _get_similarity_scores(self, query: str, snippets: List[str]) -> List[float]:
        n = len(snippets)
        queries = [query] * n
        snips = snippets.copy()

        query_embeddings = self.model.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        snip_embeddings = self.model.encode(
            snips,
            convert_to_tensor=True,
            show_progress_bar=False,
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


@CacheMemory.cache
def cached_bing_search_request(endpoint, headers, params):
    for _ in range(3):
        response = requests.get(endpoint, headers=headers, params=params)
        if not response.ok and (
            response.status_code >= 400 and response.status_code < 500
        ):
            dspy.logger.error(
                f"Bing Search HTTP error: {response.status_code}, {response.text}"
            )
            headers["Cache-Control"] = "no-cache"
            dspy.logger.error(f"req.url: {endpoint}")
        else:
            break

    return response


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
def cached_bing_search_request_wrapped(endpoint, headers, params):
    return cached_bing_search_request(endpoint, headers, params)


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
            if not response.ok and (
                response.status_code >= 400 and response.status_code < 500
            ):
                dspy.logger.error(
                    f"Brave Search HTTP error: {response.status_code}, {response.text}"
                )
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
