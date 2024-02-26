"""
Retriever model for chromadb
"""

from typing import Optional, List, Union
import openai
import dspy
import backoff
from dsp.utils import dotdict

try:
    import openai.error
    ERRORS = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError)
except Exception:
    ERRORS = (openai.RateLimitError, openai.APIError)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    from chromadb.api.types import (
        Embeddable,
        EmbeddingFunction
    )
    import chromadb.utils.embedding_functions as ef
except ImportError:
    chromadb = None

if chromadb is None:
    raise ImportError(
        "The chromadb library is required to use ChromadbRM. Install it with `pip install dspy-ai[chromadb]`"
    )


class ChromadbRM(dspy.Retrieve):
    """
    A retrieval module that uses chromadb to return the top passages for a given query.

    Assumes that the chromadb index has been created and populated with the following metadata:
        - documents: The text of the passage

    Args:
        collection_name (str): chromadb collection name
        persist_directory (str): chromadb persist directory
        embedding_function (Optional[EmbeddingFunction[Embeddable]]): Optional function to use to embed documents. Defaults to DefaultEmbeddingFunction.
        k (int, optional): The number of top passages to retrieve. Defaults to 7.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = ChromadbRM('collection_name', 'db_path')
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
        ] = None,
        k: int = 7,
    ):
        self._init_chromadb(collection_name, persist_directory)
        self.ef = embedding_function or self._chromadb_collection.embedding_function

        super().__init__(k=k)

    def _init_chromadb(
        self,
        collection_name: str,
        persist_directory: str,
    ) -> chromadb.Collection:
        """Initialize chromadb and return the loaded index.

        Args:
            collection_name (str): chromadb collection name
            persist_directory (str): chromadb persist directory


        Returns:
        """

        self._chromadb_client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                is_persistent=True,
            )
        )
        self._chromadb_collection = self._chromadb_client.get_or_create_collection(
            name=collection_name,
        )

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=15,
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
        self, query_or_queries: Union[str, List[str]], k: Optional[int] = None
    ) -> dspy.Prediction:
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
            query_embeddings=embeddings, n_results=k
        )

        passages = [dotdict({"long_text": x}) for x in results["documents"][0]]

        return passages
