import json
import os
from collections import defaultdict
from typing import List, Optional, Union

import requests

import dspy
from dspy.dsp.utils import dotdict

START_SNIPPET = "<%START%>"
END_SNIPPET = "<%END%>"

def remove_snippet(s: str) -> str:
    return s.replace(START_SNIPPET, "").replace(END_SNIPPET, "")

class VectaraRM(dspy.Retrieve):
    """
    A retrieval module that uses Vectara to return the top passages for a given query.

    Assumes that a Vectara corpora have been created and populated with the following payload:
        - document: The text of the passage

    Args:
        vectara_customer_id (str): Vectara Customer ID. defaults to VECTARA_CUSTOMER_ID environment variable
        vectara_corpus_id (str): Vectara Corpus ID. defaults to VECTARA_CORPUS_ID environment variable
        vectara_api_key (str): Vectara API Key. defaults to VECTARA_API_KEY environment variable
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.

    Examples:
        Below is a code snippet that shows how to use Vectara as the default retriver:
        ```python
        from vectara_client import vectaraClient

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = vectaraRM("<VECTARA_CUSTOMER_ID>", "<VECTARA_CORPUS_ID>", "<VECTARA_API_KEY>")
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Vectara in the forward() function of a module
        ```python
        self.retrieve = vectaraRM("<VECTARA_CUSTOMER_ID>", "<VECTARA_CORPUS_ID>", "<VECTARA_API_KEY>", k=num_passages)
        ```
    """

    def __init__(
        self,
        vectara_customer_id: Optional[str] = None,
        vectara_corpus_id: Optional[str] = None,
        vectara_api_key: Optional[str] = None,
        k: int = 5,
    ):
        if vectara_customer_id is None:
            vectara_customer_id = os.environ.get("VECTARA_CUSTOMER_ID", "")
        if vectara_corpus_id is None:
            vectara_corpus_id = os.environ.get("VECTARA_CORPUS_ID", "")
        if vectara_api_key is None:
            vectara_api_key = os.environ.get("VECTARA_API_KEY", "")

        self._vectara_customer_id = vectara_customer_id
        self._vectara_corpus_id = vectara_corpus_id
        self._vectara_api_key = vectara_api_key
        self._n_sentences_before = self._n_sentences_after = 2
        self._vectara_timeout = 120
        super().__init__(k=k)

    def _vectara_query(
        self,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """Query Vectara index to get for top k matching passages.
        Args:
            query: query string
        """
        # If multiple corpus ids are provided (comma-separated), create a list of corpus keys
        # otherwise by default, the `split(',')` is a no-op so retains the single corpus id
        corpus_key = [
            {
                "customerId": self._vectara_customer_id,
                "corpusId": corpus_id,
                "lexicalInterpolationConfig": {"lambda": 0.025 },
            } for corpus_id in self._vectara_corpus_id.split(',')
        ]

        data = {
            "query": [
                {
                    "query": query,
                    "start": 0,
                    "numResults": limit,
                    "contextConfig": {
                        "sentencesBefore": self._n_sentences_before,
                        "sentencesAfter": self._n_sentences_after,
                        "startTag": START_SNIPPET,
                        "endTag": END_SNIPPET,
                    },
                    "corpusKey": corpus_key,
                },
            ],
        }

        headers = {
            "x-api-key": self._vectara_api_key,
            "customer-id": self._vectara_customer_id,
            "Content-Type": "application/json",
            "X-Source": "dspy",
        }

        response = requests.post(
            headers=headers,
            url="https://api.vectara.io/v1/query",
            data=json.dumps(data),
            timeout=self._vectara_timeout,
        )

        if response.status_code != 200:
            print(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return []

        result = response.json()
        responses = result["responseSet"][0]["response"]

        res = [
            {
                "text": remove_snippet(x["text"]),
                "score": x["score"],
            } for x in responses
        ]
        return res
    
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction:
        """Search with Vectara for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries        
        k = k if k is not None else self.k

        all_res = []
        limit = 3*k if len(queries) > 1 else k
        for query in queries:
            _res = self._vectara_query(query, limit=limit)
            all_res.append(_res)

        passages = defaultdict(float)
        for res_list in all_res:
            for res in res_list:
                passages[res["text"]] += res["score"]
        sorted_passages = sorted(
            passages.items(), key=lambda x: x[1], reverse=True)[:k]

        return [dotdict({"long_text": passage}) for passage, _ in sorted_passages]
    