"""
Retriever model for chromadb
"""

from typing import Optional, List, Union
import openai
import dspy
import backoff
from dsp.utils import dotdict

try:
    import chromadb
    from chromadb.config import Settings

    # from chromadb.utils import embedding_functions
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
        - text: The text of the passage

    Args:
        collection_name (str): chromadb collection name
        persist_directory (str): chromadb persist directory
        openai_embed_model (str, optional): The OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        openai_api_key (str, optional): The API key for OpenAI. Defaults to None.
        openai_org (str, optional): The organization for OpenAI. Defaults to None.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriver:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = ChromadbRM('collection_name', 'db_path', openai.api_key, 'azure')
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = ChromadbRM(k=num_passages)
        ```
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        openai_embed_model: str = "text-embedding-ada-002",
        openai_api_provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_api_type: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        openai_api_version: Optional[str] = None,
        k: int = 7,
    ):
        # self._openai_embed_model = embedding_functions.OpenAIEmbeddingFunction(
        #         model_name=openai_embed_model
        # )

        self._openai_embed_model = openai_embed_model

        self._init_chromadb(collection_name, persist_directory)

        # If not provided, defaults to env vars OPENAI_API_KEY and OPENAI_ORGANIZATION
        if openai_api_key:
            openai.api_key = openai_api_key
        if openai_api_type:
            openai.api_type = openai_api_type
        if openai_api_base:
            openai.api_base = openai_api_base
        if openai_api_version:
            openai.api_version = openai_api_version
        if openai_api_provider:
            self._openai_api_provider = openai_api_provider

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
        (openai.error.RateLimitError, openai.error.ServiceUnavailableError),
        max_time=15,
    )
    def _get_embeddings(self, queries: List[str]) -> List[List[float]]:
        """Return query vector after creating embedding using OpenAI

        Args:
            queries (list): List of query strings to embed.

        Returns:
            List[List[float]]: List of embeddings corresponding to each query.
        """

        if self._openai_api_provider == "azure":
            model_args = {
                "engine": self._openai_embed_model,
                "deployment_id": self._openai_embed_model,
                "api_version": openai.api_version,
                "api_base": openai.api_base,
            }
            embedding = openai.Embedding.create(
                input=queries,
                model=self._openai_embed_model,
                **model_args,
                api_provider=self._openai_api_provider
            )
        else:
            embedding = openai.Embedding.create(
                input=queries, model=self._openai_embed_model
            )
        return [embedding["embedding"] for embedding in embedding["data"]]

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

        # 'metadatas': [[{'page': 30, 'source': "..."}, {...}]]
        passages = [dotdict({"long_text": x}) for x in results["documents"][0]]
        print(passages)

        return passages
