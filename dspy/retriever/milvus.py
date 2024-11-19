"""
Retriever model for Milvus or Zilliz Cloud
"""

from typing import List, Optional, Any

from dspy.retriever import Retriever
from dspy.clients.embedding import Embedding
from dspy.primitives.prediction import Prediction

try:
    from pymilvus import MilvusClient
except ImportError:
    raise ImportError(
        "The pymilvus library is required to use Milvus. Install it with `pip install dspy-ai[milvus]`",
    )

class Milvus(Retriever):
    """
    A retrieval module that uses Milvus to return passages for a given query.

    Assumes that a Milvus collection has been created and populated with the following field:
        - text: The text of the passage

    Args:
        collection_name (str): The name of the Milvus collection to query against.
        uri (str, optional): The Milvus connection URI. Defaults to "http://localhost:19530".
        token (str, optional): The Milvus connection token. Defaults to None.
        db_name (str, optional): The Milvus database name. Defaults to "default".
        embedder (dspy.Embedding): An instance of `dspy.Embedding` to compute embeddings.
        k (int, optional): Number of top passages to retrieve. Defaults to 5.
        callbacks (Optional[List[Any]]): List of callback functions.
        cache (bool, optional): Enable retrieval caching. Disabled by default.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        import dspy
        from dspy.retriever.milvus import Milvus

        # Create an Embedding instance
        embedder = dspy.Embedding(embedding_model="openai/text-embedding-3-small")

        retriever = Milvus(
            collection_name="<YOUR_COLLECTION_NAME>",
            uri="<YOUR_MILVUS_URI>",
            token="<YOUR_MILVUS_TOKEN>",
            embedder=embedder,
            k=3
        )
        results = retriever(query).passages
        print(results)
        ```
    """

    def __init__(
        self,
        collection_name: str,
        uri: str = "http://localhost:19530",
        token: Optional[str] = None,
        db_name: str = "default",
        embedder: Embedding = None,
        k: int = 5,
        callbacks: Optional[List[Any]] = None,
        cache: bool = False,
    ):
        if embedder is not None and not isinstance(embedder, Embedding):
            raise ValueError("If provided, the embedder must be of type `dspy.Embedding`.")
        super().__init__(embedder=embedder, k=k, callbacks=callbacks, cache=cache)

        self.milvus_client = MilvusClient(uri=uri, token=token, db_name=db_name)

        # Check if collection exists
        if collection_name not in self.milvus_client.list_collections():
            raise AttributeError(f"Milvus collection not found: {collection_name}")
        self.collection_name = collection_name

    def forward(self, query: str, k: Optional[int] = None) -> Prediction:
        """
        Retrieve passages from Milvus that are relevant to the specified query.

        Args:
            query (str): The query text for which to retrieve relevant passages.
            k (Optional[int]): The number of passages to retrieve. If None, defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k or self.k
        query_embedding = self.embedder([query])[0]

        # Milvus expects embeddings as lists
        query_embedding = query_embedding.tolist()

        milvus_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            output_fields=["text"],
            limit=k,
        )

        results = []
        for res in milvus_res:
            for r in res:
                text = r["entity"]["text"]
                doc_id = r["id"]
                distance = r["distance"]
                results.append((text, doc_id, distance))

        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)[:k]
        passages = [x[0] for x in sorted_results]
        doc_ids = [x[1] for x in sorted_results]
        distances = [x[2] for x in sorted_results]

        return Prediction(passages=passages, doc_ids=doc_ids, scores=distances)
    