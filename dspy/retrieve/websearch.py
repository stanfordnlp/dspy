from typing import Any, Dict, List, Optional
import os
from sentence_transformers import SentenceTransformer
import dspy
import requests
from collections import Counter
import torch.nn.functional as F


class BingSearch(dspy.Retrieve):
    EMBEDDING_MODEL = "avsolatorio/GIST-small-Embedding-v0"
    MAX_EMB_SEQ_LEN = 512
    DEFAULT_SEARCH_COUNT = 10

    def __init__(
        self, api_key: Optional[str] = None, endpoint: Optional[str] = None
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
            queries, convert_to_tensor=True, show_progress_bar=False
        )
        snip_embeddings = self.model.encode(
            snips, convert_to_tensor=True, show_progress_bar=False
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

