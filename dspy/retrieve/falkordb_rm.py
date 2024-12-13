import os
from typing import List, Optional, Union
import string
import random

import backoff
from openai import (
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)

from dspy import Retrieve, Prediction
from dspy.dsp.utils.settings import settings
from dspy.dsp.utils import dotdict

try:
    import falkordb
except ImportError:
    raise ImportError(
        "Please install the falkordb package by running `pip install dspy-ai[falkordb]`"
    )
import redis.exceptions


def generate_random_string(length: int) -> str:
    characters = string.ascii_letters
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


class Embedder:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Environment variable OPENAI_API_KEY must be set to"
                    "use openai as embedding provider"
                )
            self.client = OpenAI()
            self.model = model
        else:
            raise ValueError(f"Unsupported provider: {provider}")

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
    def __call__(self, queries: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(queries, str):
            queries = [queries]

        if self.provider == "openai":
            embedding = self.client.embeddings.create(input=queries, model=self.model)
            return [result.embedding for result in embedding.data]


DEFAULT_INDEX_QUERY = "CALL db.idx.vector.queryNodes($node_label, $embedding_node_property, $k, vecf32($embedding)) YIELD node, score "


class FalkordbRM(Retrieve):
    """
    Implements a retriever that utilizes FalkorDB for retrieving passages.
    This class manages a connection to a FalkorDB database using official FalkorDB Python drivers and requires
    the database credentials. That is, if using a local FalkorDB session, host and port else if using a FalkorDB cloud session,
    host, port, username, and password to be set as environment variables and optionally the database name.
    Additionally, it utilizes an embedding provider (defaulting to OpenAI's services) to compute query embeddings,
    which are then used to find the most relevant nodes in the FalkorDB graph based on the specified node property or custom retrieval query.

    Returns a list of passages in the form of `dspy.Prediction` objects

    Args:
        Args:
        node_label (str): The label of the node in the FalkorDB database to query against
        text_node_property (str): The property of the node containing the text.
        embedding_node_property (List[float]): The property of the node containing the embeddings.
        k (Optional[int]): The default number of top passages to retrieve. Defaults to 5.
        retrieval_query (Optional[str]): Custom Cypher query for retrieving passages.
        embedding_provider (str): The provider of the embedding service. Defaults to "openai".
        embedding_model (str): The model identifier for generating embeddings. Defaults to "text-embedding-ada-002".

    Examples:
        Below is a code snippet showcasing how to initialize FalkordbRM with environment variables for the database connection and OpenAI as the embedding provider:

        ```python
        import os

        import dspy
        import openai

        os.environ["FALKORDB_HOST"] = "localhost"
        os.environ["FALORDB_PORT"] = "6379"
        os.environ["OPENAI_API_KEY"] = "sk-" (Only if using openai as embedding's provider)

        # Uncomment and set the following if you are using FalkorDB cloud
        # os.environ["FALKORDB_USERNAME"] = "falkordb"
        # os.environ["FALKORDB_PASSWORD"] = "password"


        falkordb_retriever = FalkordbRM(
            node_label="myIndex",
            text_node_property="text",
            k=10,
            embedding_provider="openai",
            embedding_model="text-embedding-ada-002",
        )

        dspy.settings.configure(rm=falkordb_retriever)
        ```

        In this example, `FalkordbRM` is configured to retrieve nodes based on the "text" property from an index on a node labeled "myIndex",
        using embeddings computed by OpenAI's "text-embedding-ada-002" model.
    """

    def __init__(
        self,
        node_label: str,
        text_node_property: str = None,
        embedding_node_property: str = None,
        k: int = 5,
        retrieval_query: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
    ):
        super().__init__(k=k)
        self.node_label = node_label
        self.username = os.getenv("FALKORDB_USERNAME", None)
        self.password = os.getenv("FALKORDB_PASSWORD", None)
        self.host = os.getenv("FALKORDB_HOST", "localhost")
        self.port = int(os.getenv("FALKORDB_PORT", 6379))

        self.database = os.getenv("FALKORDB_DATABASE", generate_random_string(4))
        self.k = k
        self.retrieval_query = retrieval_query
        self.text_node_property = text_node_property
        self.embedding_node_property = embedding_node_property
        if not self.text_node_property and not self.retrieval_query:
            raise ValueError(
                "Either `text_node_property` or `retrieval_query` must be set"
            )
        if not embedding_node_property:
            raise ValueError("`embedding_node_property` must be set")
        try:
            self.driver = falkordb.FalkorDB(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
            ).select_graph(self.database)

        except (
            redis.exceptions.ConnectionError,
            redis.exceptions.AuthenticationError,
        ) as e:
            raise ConnectionError("Failed to connect to FalkorDB database") from e

        self.embedder = Embedder(provider=embedding_provider, model=embedding_model)

    def forward(
        self, query_or_queries: Union[str, List[str]], k: Optional[int]
    ) -> Prediction:
        if not isinstance(query_or_queries, list):
            query_or_queries = [query_or_queries]
        query_vectors = self.embedder(query_or_queries)
        contents = []
        retrieval_query = (
            self.retrieval_query
            or f"RETURN node.{self.text_node_property} AS text, score"
        )
        if not k:
            k = self.k

        for vector in query_vectors:
            params = {
                "embedding": vector,
                "node_label": self.node_label,
                "text_node_property": self.text_node_property,
                "embedding_node_property": self.embedding_node_property,
                "k": k,
            }
            try:
                records = self.driver.query(
                    DEFAULT_INDEX_QUERY + retrieval_query,
                    params=params,
                ).result_set
            except Exception as e:
                if "Invalid arguments" in str(e):
                    raise ValueError(
                        f"There is no vector index on node label, {self.node_label}"
                        f" and node property, {self.embedding_node_property}"
                    )
            contents.extend(
                [
                    {"passage": dotdict({"long_text": r[1]}), "score": r[0]}
                    for r in records
                ]
            )
        sorted_passages = sorted(
            contents,
            key=lambda x: x["score"],
            reverse=True,
        )[:k]
        return [el["passage"] for el in sorted_passages]
