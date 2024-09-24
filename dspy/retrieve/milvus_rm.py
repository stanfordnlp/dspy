"""
Retriever model for Milvus or Zilliz Cloud
"""

from typing import Callable, List, Optional, Union

import dspy
from dsp.utils import dotdict

try:
    from pymilvus import MilvusClient
except ImportError:
    raise ImportError(
        "The pymilvus library is required to use MilvusRM. Install it with `pip install dspy-ai[milvus]`",
    )


def openai_embedding_function(texts: List[str]):
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [x.embedding for x in response.data]


class MilvusRM(dspy.Retrieve):
    """
    A retrieval module that uses Milvus to return passages for a given query.

    Assumes that a Milvus collection has been created and populated with the following field:
        - text: The text of the passage

    Args:
        collection_name (str): The name of the Milvus collection to query against.
        uri (str, optional): The Milvus connection uri. Defaults to "http://localhost:19530".
        token (str, optional): The Milvus connection token. Defaults to None.
        db_name (str, optional): The Milvus database name. Defaults to "default".
        embedding_function (callable, optional): The function to convert a list of text to embeddings. 
            The embedding function should take a list of text strings as input and output a list of embeddings.
            Defaults to None. By default, it will get OpenAI client by the environment variable OPENAI_API_KEY 
            and use OpenAI's embedding model "text-embedding-3-small" with the default dimension.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = MilvusRM(
            collection_name="<YOUR_COLLECTION_NAME>",
            uri="<YOUR_MILVUS_URI>",
            token="<YOUR_MILVUS_TOKEN>"
            )
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = MilvusRM(k=num_passages)
        ```
    """

    def __init__(
        self,
        collection_name: str,
        uri: Optional[str] = "http://localhost:19530",
        token: Optional[str] = None,
        db_name: Optional[str] = "default",
        embedding_function: Optional[Callable] = None,
        k: int = 3,
    ):
        self.milvus_client = MilvusClient(uri=uri, token=token, db_name=db_name)

        # check collection existence
        if collection_name not in self.milvus_client.list_collections():
            raise AttributeError(f"Milvus collection not found: {collection_name}")
        self.collection_name = collection_name

        self.embedding_function = embedding_function or openai_embedding_function
        self.k = k

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        if isinstance(query_or_queries, str):
            query_or_queries = [query_or_queries]
        query_embeddings = self.embedding_function(query_or_queries)

        k = k or self.k

        milvus_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_embeddings,
            output_fields=["text"],
            limit=k,
        )
        passage_scores = {}
        for res in milvus_res:
            for r in res:
                passage_scores[r["entity"]["text"]] = r["distance"]
        sorted_passages = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True,
        )[:k]
        return [dotdict({"long_text": passage}) for passage, _ in sorted_passages]