from typing import List, Union, Any
import dspy
import os
import openai
import backoff

try:
    from pymongo import MongoClient
    from pymongo.errors import (
        ConnectionFailure,
        ConfigurationError,
        ServerSelectionTimeoutError,
        InvalidURI,
        OperationFailure,
    )
except ImportError:
    raise ImportError(
        "Please install the pymongo package by running `pip install dspy-ai[mongodb]`"
    )


def build_vector_search_pipeline(
    index_name: str, query_vector: List[float], num_candidates: int, limit: int
) -> List[dict[str, Any]]:
    return [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        {"$project": {"_id": 0, "text": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]


class Embedder:
    def __init__(self, provider: str, model: str):
        if provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("Environment variable OPENAI_API_KEY must be set")
            self.client = openai
            self.model = model

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIError,
        ),
        max_time=15,
    )
    def __call__(self, queries) -> Any:
        embedding = self.client.Embedding.create(input=queries, model=self.model)
        return [embedding["embedding"] for embedding in embedding["data"]]


class MongoDBAtlasRM(dspy.Retrieve):
    def __init__(
        self,
        db_name: str,
        collection_name: str,
        index_name: str,
        k: int = 5,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
    ):
        super().__init__(k=k)
        self.db_name = db_name
        self.collection_name = collection_name
        self.index_name = index_name
        self.username = os.getenv("ATLAS_USERNAME")
        self.password = os.getenv("ATLAS_PASSWORD")
        self.cluster_url = os.getenv("ATLAS_CLUSTER_URL")
        if not self.username:
            raise ValueError("Environment variable ATLAS_USERNAME must be set")
        if not self.password:
            raise ValueError("Environment variable ATLAS_PASSWORD must be set")
        if not self.cluster_url:
            raise ValueError("Environment variable ATLAS_CLUSTER_URL must be set")
        try:
            self.client = MongoClient(
                f"mongodb+srv://{self.username}:{self.password}@{self.cluster_url}/{self.db_name}"
                "?retryWrites=true&w=majority"
            )
        except (
            InvalidURI,
            ConfigurationError,
            ConnectionFailure,
            ServerSelectionTimeoutError,
            OperationFailure,
        ) as e:
            raise ConnectionError("Failed to connect to MongoDB Atlas") from e

        self.embedder = Embedder(provider=embedding_provider, model=embedding_model)

    def forward(self, query_or_queries: Union[str, List[str]]) -> dspy.Prediction:
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        query_vector = self.embedder(queries)
        pipeline = build_vector_search_pipeline(
            index_name=self.index_name,
            query_vector=query_vector[0],
            num_candidates=self.k * 10,
            limit=self.k,
        )
        contents = self.client[self.db_name][self.collection_name].aggregate(pipeline)
        return dspy.Prediction(passages=list(contents))
