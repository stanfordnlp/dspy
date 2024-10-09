"""
Retriever model for chromadb
"""

from typing import List, Optional, Union

import backoff
import openai

from dspy import Retrieve, Prediction
from dsp.utils.settings import settings
from dsp.utils import dotdict

try:
    import openai.error
    ERRORS = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError)
except Exception:
    ERRORS = (openai.RateLimitError, openai.APIError)

try:
    import chromadb
    import chromadb.utils.embedding_functions as ef
    from chromadb.api.types import (
        Embeddable,
        EmbeddingFunction,
    )
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError(
        "The chromadb library is required to use ChromadbRM. Install it with `pip install dspy-ai[chromadb]`",
    )


class ChromadbRM(Retrieve):
    """
    A retrieval module that uses chromadb to return the top passages for a given query.

    Assumes that the chromadb index has been created and populated with the following metadata:
        - documents: The text of the passage

    Args:
        collection_name (str): chromadb collection name
        persist_directory (str): chromadb persist directory
        embedding_function (Optional[EmbeddingFunction[Embeddable]]): Optional function to use to embed documents. Defaults to DefaultEmbeddingFunction.
        k (int, optional): The number of top passages to retrieve. Defaults to 7.
        client(Optional[chromadb.Client]): Optional chromadb client provided by user, default to None

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        # using default chromadb client
        retriever_model = ChromadbRM('collection_name', 'db_path')
        dspy.settings.configure(lm=llm, rm=retriever_model)
        # to test the retriever with "my query"
        retriever_model("my query")
        ```

        Use provided chromadb client
        ```python
        import chromadb
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        # say you have a chromadb running on a different port
        client = chromadb.HttpClient(host='localhost', port=8889)
        retriever_model = ChromadbRM('collection_name', 'db_path', client=client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        # to test the retriever with "my query"
        retriever_model("my query")
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = ChromadbRM('collection_name', 'db_path', k=num_passages)
        ```
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_function: Optional[
            EmbeddingFunction[Embeddable]
        ] = ef.DefaultEmbeddingFunction(),
        client: Optional[chromadb.Client] = None,
        k: int = 7,
    ):
        self._init_chromadb(collection_name, persist_directory, client=client)
        self.ef = embedding_function

        super().__init__(k=k)

    def _init_chromadb(
        self,
        collection_name: str,
        persist_directory: str,
        client: Optional[chromadb.Client] = None,
    ) -> chromadb.Collection:
        """Initialize chromadb and return the loaded index.

        Args:
            collection_name (str): chromadb collection name
            persist_directory (str): chromadb persist directory
            client (chromadb.Client): chromadb client provided by user

        Returns: collection per collection_name
        """

        if client:
            self._chromadb_client = client
        else:
            self._chromadb_client = chromadb.Client(
                Settings(
                    persist_directory=persist_directory,
                    is_persistent=True,
                ),
        )
        self._chromadb_collection = self._chromadb_client.get_or_create_collection(
            name=collection_name,
        )

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
    )
    def _get_embeddings(self, queries: List[str]) -> List[List[float]]:
        """Return query vector after creating embedding using OpenAI

        Args:
            queries (list): List of query strings to embed.

        Returns:
            List[List[float]]: List of embeddings corresponding to each query.
        """
        return self.ef(queries)

    def forward(
        self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs,
    ) -> Prediction:
        """Search with db for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries
        embeddings = self._get_embeddings(queries)

        k = self.k if k is None else k
        results = self._chromadb_collection.query(
            query_embeddings=embeddings, n_results=k,**kwargs,
        )

        zipped_results = zip(
            results["ids"][0], 
            results["distances"][0], 
            results["documents"][0], 
            results["metadatas"][0])
        results = [dotdict({"id": id, "score": dist, "long_text": doc, "metadatas": meta }) for id, dist, doc, meta in zipped_results]
        return results
