from collections import defaultdict
from typing import List, Union, Optional
import dspy
from dsp.utils import dotdict

try:
    from qdrant_client import QdrantClient
    import fastembed
except ImportError:
    raise ImportError(
        "The 'qdrant' extra is required to use QdrantRM. Install it with `pip install dspy-ai[qdrant]`"
    )


class QdrantRM(dspy.Retrieve):
    """
    A retrieval module that uses Qdrant to return the top passages for a given query.

    Assumes that a Qdrant collection has been created and populated with the following payload:
        - document: The text of the passage

    Args:
        qdrant_collection_name (str): The name of the Qdrant collection.
        qdrant_client (QdrantClient): A QdrantClient instance.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.

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
        self.retrieve = QdrantRM("my_collection_name", qdrant_client=qdrant_client, k=num_passages)
        ```
    """

    def __init__(
        self,
        qdrant_collection_name: str,
        qdrant_client: QdrantClient,
        k: int = 3,
    ):
        self._qdrant_collection_name = qdrant_collection_name
        self._qdrant_client = qdrant_client

        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction:
        """Search with Qdrant for self.k top passages for query

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
        batch_results = self._qdrant_client.query_batch(
            self._qdrant_collection_name, query_texts=queries, limit=k)

        passages_scores = defaultdict(float)
        for batch in batch_results:
            for result in batch:
                # If a passage is returned multiple times, the score is accumulated.
                passages_scores[result.document] += result.score

        # Sort passages by their accumulated scores in descending order
        sorted_passages = sorted(
            passages_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Wrap each sorted passage in a dotdict with 'long_text'
        passages = [dotdict({"long_text": passage}) for passage, _ in sorted_passages]

        return passages
