"""
Retriever model for Pinecone
Author: Dhar Rawal (@drawal1)
"""

from typing import Optional, List, Union
import openai
import dspy
import backoff

try:
    import pinecone
except ImportError:
    pinecone = None

if pinecone is None:
    raise ImportError(
        "The pinecone library is required to use PineconeRM. Install it with `pip install dspy-ai[pinecone]`"
    )


class PineconeRM(dspy.Retrieve):
    """
    A retrieval module that uses Pinecone to return the top passages for a given query.

    Assumes that the Pinecone index has been created and populated with the following metadata:
        - text: The text of the passage

    Args:
        pinecone_index_name (str): The name of the Pinecone index to query against.
        pinecone_api_key (str, optional): The Pinecone API key. Defaults to None.
        pinecone_env (str, optional): The Pinecone environment. Defaults to None.
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
        retriever_model = PineconeRM(openai.api_key)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = PineconeRM(k=num_passages)
        ```
    """

    def __init__(
        self,
        pinecone_index_name: str,
        pinecone_api_key: Optional[str] = None,
        pinecone_env: Optional[str] = None,
        openai_embed_model: str = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        k: int = 3,
    ):
        self._openai_embed_model = openai_embed_model
        self._pinecone_index = self._init_pinecone(
            pinecone_index_name, pinecone_api_key, pinecone_env
        )

        # If not provided, defaults to env vars OPENAI_API_KEY and OPENAI_ORGANIZATION
        if openai_api_key:
            openai.api_key = openai_api_key
        if openai_org:
            openai.organization = openai_org

        super().__init__(k=k)

    def _init_pinecone(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> pinecone.Index:
        """Initialize pinecone and return the loaded index.

        Args:
            index_name (str): The name of the index to load.
            api_key (str, optional): The Pinecone API key, defaults to env var PINECONE_API_KEY if not provided.
            environment (str, optional): The environment (ie. `us-west1-gcp`. Defaults to env PINECONE_ENVIRONMENT.

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

        return pinecone.Index(index_name)

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
        embedding = openai.Embedding.create(
            input=queries, model=self._openai_embed_model
        )
        return [embedding["embedding"] for embedding in embedding["data"]]

    def forward(self, query_or_queries: Union[str, List[str]]) -> dspy.Prediction:
        """Search with pinecone for self.k top passages for query

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

        # For single query, just look up the top k passages
        if len(queries) == 1:
            results_dict = self._pinecone_index.query(
                embeddings[0], top_k=self.k, include_metadata=True
            )

            # Sort results by score
            sorted_results = sorted(
                results_dict["matches"], key=lambda x: x["score"], reverse=True
            )
            passages = [result["metadata"]["text"] for result in sorted_results]
            passages = [dotdict({"long_text": passage for passage in passages})]
            return dspy.Prediction(passages=passages)

        # For multiple queries, query each and return the highest scoring passages
        # If a passage is returned multiple times, the score is accumulated. For this reason we increase top_k by 3x
        passage_scores = {}
        for embedding in embeddings:
            results_dict = self._pinecone_index.query(
                embedding, top_k=self.k * 3, include_metadata=True
            )
            for result in results_dict["matches"]:
                passage_scores[result["metadata"]["text"]] = (
                    passage_scores.get(result["metadata"]["text"], 0.0)
                    + result["score"]
                )

        sorted_passages = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True
        )[: self.k]
        return dspy.Prediction(passages=[dotdict({"long_text": passage}) for passage, _ in sorted_passages])
