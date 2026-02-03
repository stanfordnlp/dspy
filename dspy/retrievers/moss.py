import asyncio

from dspy.dsp.utils import dotdict
from dspy.primitives.prediction import Prediction
from dspy.retrievers.retrieve import Retrieve

try:
    from inferedge_moss import DocumentInfo, MossClient, QueryOptions
except ImportError as err:
    raise ImportError(
        "The 'moss' extra is required to use MossRM. Install it with `pip install dspy-ai[moss]`",
    ) from err


class MossRM(Retrieve):
    """A retrieval module that uses Moss (InferEdge) to return the top passages for a given query.

    Args:
        index_name (str): The name of the Moss index.
        moss_client (MossClient): An instance of the Moss client.
        k (int, optional): The default number of top passages to retrieve. Default to 3.

    Examples:
        Below is a code snippet that shows how to use Moss as the default retriever:
        ```python
        from inferedge_moss import MossClient
        import dspy

        moss_client = MossClient("your-project-id", "your-project-key")
        retriever_model = MossRM("my_index_name", moss_client=moss_client)
        dspy.configure(rm=retriever_model)

        retrieve = dspy.Retrieve(k=1)
        topK_passages = retrieve("what are the stages in planning, sanctioning and execution of public works").passages
        ```
    """

    def __init__(
        self,
        index_name: str,
        moss_client: MossClient,
        k: int = 3,
        alpha: float = 0.5,
    ):
        self._index_name = index_name
        self._moss_client = moss_client
        self._alpha = alpha

        super().__init__(k=k)

    def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs) -> Prediction:
        """Search with Moss for self.k top passages for query or queries.

        Args:
            query_or_queries (Union[str, list[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
            kwargs : Additional arguments for Moss client.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages = []

        for query in queries:
            options = QueryOptions(top_k=k, alpha=self._alpha, **kwargs)
            # Since MossClient methods are async, we use asyncio.run to call them synchronously.
            # This assumes the loop is not already running, which is typical for DSPy RM calls.
            result = asyncio.run(self._moss_client.query(self._index_name, query, options=options))

            for doc in result.docs:
                passages.append(
                    dotdict({"long_text": doc.text, "id": doc.id, "metadata": doc.metadata, "score": doc.score})
                )

        return passages

    def get_objects(self, num_samples: int = 5) -> list[dict]:
        """Get objects from Moss."""
        # Note: Moss's get_docs might return all docs or have limits.
        # Here we attempt to fetch and return up to num_samples.
        result = asyncio.run(self._moss_client.get_docs(self._index_name))
        # result is likely a list of DocumentInfo or similar
        objects = []
        for i, doc in enumerate(result):
            if i >= num_samples:
                break
            objects.append({"id": doc.id, "text": doc.text, "metadata": doc.metadata})
        return objects

    def insert(self, new_object_properties: dict | list[dict]):
        """Insert one or more objects into Moss."""
        if isinstance(new_object_properties, dict):
            new_object_properties = [new_object_properties]

        docs = [DocumentInfo(**props) for props in new_object_properties]
        asyncio.run(self._moss_client.add_docs(self._index_name, docs))
