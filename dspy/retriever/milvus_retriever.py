"""
Retriever model for Milvus or Zilliz Cloud
"""

from typing import List, Optional, Any

import dspy
from dspy import Embedder

try:
    from pymilvus import MilvusClient
except ImportError:
    raise ImportError(
        "The pymilvus library is required to use MilvusRetriever. Install it with `pip install dspy-ai[milvus]`",
    )

class MilvusRetriever(dspy.Retriever):
    """
    A retrieval module that uses Milvus to return passages for a given query.

    Assumes that a Milvus collection has been created and populated with the following field:
        - text: The text of the passage

    Args:
        collection_name (str): The name of the Milvus collection to query against.
        uri (str, optional): The Milvus connection URI. Defaults to "http://localhost:19530".
        token (str, optional): The Milvus connection token. Defaults to None.
        db_name (str, optional): The Milvus database name. Defaults to "default".
        embedder (dspy.Embedder): An instance of `dspy.Embedder` to compute embeddings.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.
        callbacks (Optional[List[Any]]): A list of callback functions.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        import dspy
        from dspy.retriever.milvus_retriever import MilvusRetriever

        # Create an Embedder instance
        embedder = dspy.Embedder(embedding_model="text-embedding-ada-002")

        retriever = MilvusRetriever(
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
        uri: Optional[str] = "http://localhost:19530",
        token: Optional[str] = None,
        db_name: Optional[str] = "default",
        embedder: Embedder = None,
        k: int = 3,
        callbacks: Optional[List[Any]] = None,
    ):
        if embedder is not None and not isinstance(embedder, dspy.Embedder):
            raise ValueError("If provided, the embedder must be of type `dspy.Embedder`.")
        super().__init__(embedder=embedder, k=k, callbacks=callbacks)

        self.milvus_client = MilvusClient(uri=uri, token=token, db_name=db_name)

        # Check if collection exists
        if collection_name not in self.milvus_client.list_collections():
            raise AttributeError(f"Milvus collection not found: {collection_name}")
        self.collection_name = collection_name

    def forward(self, query: str, k: Optional[int] = None) -> dspy.Prediction:
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

        return dspy.Prediction(passages=passages, doc_ids=doc_ids, scores=distances)