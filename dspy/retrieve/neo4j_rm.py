import os
from typing import Any, List, Optional, Union

import backoff
from openai import (
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)

from dsp.utils.settings import settings
from dsp.utils import dotdict

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import (
        AuthError,
        ServiceUnavailable,
    )
except ImportError:
    raise ImportError(
        "Please install the neo4j package by running `pip install dspy-ai[neo4j]`",
    )


class Embedder:
    def __init__(self, provider: str, model: str):
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Environment variable OPENAI_API_KEY must be set")
            self.client = OpenAI()
            self.model = model

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=settings.backoff_time,
    )
    def __call__(self, queries) -> Any:
        embedding = self.client.embeddings.create(input=queries, model=self.model)
        return [result.embedding for result in embedding.data]


DEFAULT_INDEX_QUERY = "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "


class Neo4jRM(dspy.Retrieve):
    """
    Implements a retriever that utilizes Neo4j for retrieving passages.
    This class manages a connection to a Neo4j database using official Neo4j Python drivers and requires
    the database credentials (username, password, URI, and optionally the database name) to be set as environment variables.
    Additionally, it utilizes an embedding provider (defaulting to OpenAI's services) to compute query embeddings,
    which are then used to find the most relevant nodes in the Neo4j graph based on the specified node property or custom retrieval query.

    Returns a list of passages in the form of `dspy.Prediction` objects.

    Args:
        index_name (str): The name of the vector index in the Neo4j database to query against.
        text_node_property (Optional[str]): The property of the node containing the text. Required if `retrieval_query` is not set.
        k (Optional[int]): The default number of top passages to retrieve. Defaults to 5.
        retrieval_query (Optional[str]): Custom Cypher query for retrieving passages. Required if `text_node_property` is not set.
        embedding_provider (str): The provider of the embedding service. Defaults to "openai".
        embedding_model (str): The model identifier for generating embeddings. Defaults to "text-embedding-ada-002".

    Examples:
        Below is a code snippet showcasing how to initialize Neo4jRM with environment variables for the database connection and OpenAI as the embedding provider:

        ```python
        import os

        import dspy
        import openai

        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_PASSWORD"] = "password"
        os.environ["OPENAI_API_KEY"] = "sk-"

        neo4j_retriever = Neo4jRM(
            index_name="myIndex",
            text_node_property="text",
            k=10,
            embedding_provider="openai",
            embedding_model="text-embedding-ada-002",
        )

        dspy.settings.configure(rm=neo4j_retriever)
        ```

        In this example, `Neo4jRM` is configured to retrieve nodes based on the "text" property from an index named "myIndex",
        using embeddings computed by OpenAI's "text-embedding-ada-002" model.
    """

    def __init__(
        self,
        index_name: str,
        text_node_property: str = None,
        k: int = 5,
        retrieval_query: str = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
    ):
        super().__init__(k=k)
        self.index_name = index_name
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.uri = os.getenv("NEO4J_URI")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.k = k
        self.retrieval_query = retrieval_query
        self.text_node_property = text_node_property
        if not self.username:
            raise ValueError("Environment variable NEO4J_USERNAME must be set")
        if not self.password:
            raise ValueError("Environment variable NEO4J_PASSWORD must be set")
        if not self.uri:
            raise ValueError("Environment variable NEO4J_URI must be set")
        if not self.text_node_property and not self.retrieval_query:
            raise ValueError("Either `text_node_property` or `retrieval_query` parameters must be defined")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()

        except (
            ServiceUnavailable,
            AuthError,
        ) as e:
            raise ConnectionError("Failed to connect to Neo4j database") from e

        self.embedder = Embedder(provider=embedding_provider, model=embedding_model)

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction:
        if not isinstance(query_or_queries, list):
            query_or_queries = [query_or_queries]
        query_vectors = self.embedder(query_or_queries)
        contents = []
        retrieval_query = self.retrieval_query or f"RETURN node.{self.text_node_property} AS text, score"
        for vector in query_vectors:
            records, _, _ = self.driver.execute_query(
                DEFAULT_INDEX_QUERY + retrieval_query,
                {"embedding": vector, "index": self.index_name, "k": k or self.k},
                database_=self.database,
            )
            contents.extend([{"passage": dotdict({"long_text": r["text"]}), "score": r["score"]} for r in records])
        sorted_passages = sorted(
            contents,
            key=lambda x: x["score"],
            reverse=True,
        )[: k or self.k]
        return [el["passage"] for el in sorted_passages]
