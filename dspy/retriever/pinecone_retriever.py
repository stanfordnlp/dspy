"""
Retriever model for Pinecone
Author: Dhar Rawal (@drawal1)
(modified to support `dspy.Retriever` interface)
"""

from typing import List, Optional, Any, Union

import dspy
from dspy import Embedder
from dspy.primitives.prediction import Prediction
from dsp.utils import dotdict

try:
    import pinecone
except ImportError:
    pinecone = None

if pinecone is None:
    raise ImportError(
        "The pinecone library is required to use PineconeRetriever. Install it with `pip install dspy-ai[pinecone]`",
    )


class PineconeRetriever(dspy.Retriever):
    """
    A retrieval module that uses Pinecone to return the top passages for a given query or list of queries.

    Assumes that the Pinecone index has been created and populated with the following metadata:
        - text: The text of the passage

    Args:
        pinecone_index_name (str): The name of the Pinecone index to query against.
        pinecone_api_key (str, optional): The Pinecone API key. Defaults to None.
        pinecone_env (str, optional): The Pinecone environment. Defaults to None.
        embedder (dspy.Embedder): An instance of `dspy.Embedder` to compute embeddings.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.
        callbacks (Optional[List[Any]]): A list of callback functions.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        import dspy
        from dspy.retriever.pinecone_retriever import PineconeRetriever

        # Create an Embedder instance
        embedder = dspy.Embedder(embedding_model="text-embedding-ada-002")

        retriever = PineconeRetriever(
            pinecone_index_name="<YOUR_INDEX_NAME>",
            pinecone_api_key="<YOUR_PINECONE_API_KEY>",
            pinecone_env="<YOUR_PINECONE_ENV>",
            embedder=embedder,
            k=3
        )

        results = retriever(query).passages
        print(results)
        ```
    """

    def __init__(
        self,
        pinecone_index_name: str,
        pinecone_api_key: Optional[str] = None,
        pinecone_env: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[str] = None,
        embedder: Embedder = None,
        k: int = 3,
        callbacks: Optional[List[Any]] = None,
    ):
        if embedder is None or not isinstance(embedder, dspy.Embedder):
            raise ValueError("An embedder of type `dspy.Embedder` must be provided.")
        self.embedder = embedder
        super().__init__(embedder=self.embedder, k=k, callbacks=callbacks)

        self._pinecone_index = self._init_pinecone(
            index_name=pinecone_index_name,
            api_key=pinecone_api_key,
            environment=pinecone_env,
            dimension=dimension,
            distance_metric=distance_metric,
        )

    def _init_pinecone(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[str] = None,
    ) -> pinecone.Index:
        """Initialize pinecone and return the loaded index.

        Args:
            index_name (str): The name of the index to load. If the index is not does not exist, it will be created.
            api_key (str, optional): The Pinecone API key, defaults to env var PINECONE_API_KEY if not provided.
            environment (str, optional): The environment (ie. `us-west1-gcp` or `gcp-starter`. Defaults to env PINECONE_ENVIRONMENT.

        Raises:
            ValueError: If api_key or environment is not provided and not set as an environment variable.

        Returns:
            pinecone.Index: The loaded index.
        """

        # Pinecone init overrides default if kwargs are present, so we need to exclude if None
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if environment:
            kwargs["environment"] = environment
        pinecone.init(**kwargs)

        active_indexes = pinecone.list_indexes()
        if index_name not in active_indexes:
            if dimension is None or distance_metric is None:
                raise ValueError(
                    "dimension and distance_metric must be provided since the index does not exist and needs to be created."
                )

            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=distance_metric,
            )

        return pinecone.Index(index_name)

    def forward(self, query: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        """Search with Pinecone for top k passages for the query or queries.

        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k or self.k
        queries = [query] if isinstance(query, str) else query
        queries = [q for q in queries if q]
        embeddings = self.embedder(queries)
        # For single query, just look up the top k passages
        if len(queries) == 1:
            results_dict = self._pinecone_index.query(
                embeddings[0], top_k=self.k, include_metadata=True,
            )

            # Sort results by score
            sorted_results = sorted(
                results_dict["matches"], key=lambda x: x.get("scores", 0.0), reverse=True,
            )
            passages = [result["metadata"]["text"] for result in sorted_results]
            passages = [dotdict({"long_text": passage for passage in passages})]
            return Prediction(passages=passages)

        # For multiple queries, query each and return the highest scoring passages
        # If a passage is returned multiple times, the score is accumulated. For this reason we increase top_k by 3x
        passage_scores = {}
        for embedding in embeddings:
            results_dict = self._pinecone_index.query(
                embedding, top_k=self.k * 3, include_metadata=True,
            )
            for result in results_dict["matches"]:
                passage_scores[result["metadata"]["text"]] = (
                    passage_scores.get(result["metadata"]["text"], 0.0)
                    + result["score"]
                )

        sorted_passages = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True,
        )[: self.k]
        return Prediction(passages=[dotdict({"long_text": passage}) for passage, _ in sorted_passages])