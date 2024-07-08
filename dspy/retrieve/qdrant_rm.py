from collections import defaultdict
from typing import Optional, Union

import dspy
from dsp.modules.sentence_vectorizer import BaseSentenceVectorizer, FastEmbedVectorizer
from dsp.utils import dotdict

try:
    from qdrant_client import QdrantClient, models
except ImportError as e:
    raise ImportError(
        "The 'qdrant' extra is required to use QdrantRM. Install it with `pip install dspy-ai[qdrant]`",
    ) from e


class QdrantRM(dspy.Retrieve):
    """A retrieval module that uses Qdrant to return the top passages for a given query.

    Args:
        qdrant_collection_name (str): The name of the Qdrant collection.
        qdrant_client (QdrantClient): An instance of `qdrant_client.QdrantClient`.
        k (int, optional): The default number of top passages to retrieve. Default: 3.
        document_field (str, optional): The key in the Qdrant payload with the content. Default: `"document"`.
        vectorizer (BaseSentenceVectorizer, optional): An implementation `BaseSentenceVectorizer`.
                                                           Default: `FastEmbedVectorizer`.
        vector_name (str, optional): Name of the vector in the collection. Default: The first available vector name.

    Examples:
        Below is a code snippet that shows how to use Qdrant as the default retriver:
        ```python
        from qdrant_client import QdrantClient

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        qdrant_client = QdrantClient()
        retriever_model = QdrantRM("my_collection_name", qdrant_client=qdrant_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Qdrant in the forward() function of a module
        ```python
        self.retrieve = QdrantRM(question, k=num_passages, filter=filter)
        ```
    """

    def __init__(
        self,
        qdrant_collection_name: str,
        qdrant_client: QdrantClient,
        k: int = 3,
        document_field: str = "document",
        vectorizer: Optional[BaseSentenceVectorizer] = None,
        vector_name: Optional[str] = None,
    ):
        self._collection_name = qdrant_collection_name
        self._client = qdrant_client

        self._vectorizer = vectorizer or FastEmbedVectorizer(self._client.embedding_model_name)

        self._document_field = document_field

        self._vector_name = vector_name or self._get_first_vector_name()

        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None, filter: Optional[models.Filter]=None) -> dspy.Prediction:
        """Search with Qdrant for self.k top passages for query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
            filter (Optional["Filter"]): "Look only for points which satisfies this conditions". Default: None.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries

        vectors = self._vectorizer(queries)

        # If vector_name is None
        # vector = [0.8, 0.2, 0.3...]
        # Else
        # vector = {"name": vector_name, "vector": [0.8, 0.2, 0.3...]}
        vectors = [
            vector if self._vector_name is None else {"name": self._vector_name, "vector": vector} for vector in vectors
        ]

        search_requests = [
            models.SearchRequest(
                vector=vector,
                limit=k or self.k,
                with_payload=[self._document_field],
                filter=filter,
            )
            for vector in vectors
        ]
        batch_results = self._client.search_batch(self._collection_name, requests=search_requests)

        passages_scores = defaultdict(float)
        for batch in batch_results:
            for result in batch:
                # If a passage is returned multiple times, the score is accumulated.
                document = result.payload.get(self._document_field)
                passages_scores[document] += result.score

        # Sort passages by their accumulated scores in descending order
        sorted_passages = sorted(passages_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Wrap each sorted passage in a dotdict with 'long_text'
        return [dotdict({"long_text": passage}) for passage, _ in sorted_passages]

    def _get_first_vector_name(self) -> Optional[str]:
        vectors = self._client.get_collection(self._collection_name).config.params.vectors

        if not isinstance(vectors, dict):
            # The collection only has the default, unnamed vector
            return None

        first_vector_name = list(vectors.keys())[0]

        # The collection has multiple vectors. Could also include the falsy unnamed vector - Empty string("")
        return first_vector_name or None
