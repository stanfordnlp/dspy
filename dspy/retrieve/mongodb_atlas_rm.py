import os
from typing import Any, List

import backoff
from openai import (
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)

from dsp.utils.settings import settings

try:
    from pymongo import MongoClient
    from pymongo.errors import (
        ConfigurationError,
        ConnectionFailure,
        InvalidURI,
        OperationFailure,
        ServerSelectionTimeoutError,
    )
except ImportError:
    raise ImportError(
        "Please install the pymongo package by running `pip install dspy-ai[mongodb]`",
    )


def build_vector_search_pipeline(
    index_name: str, query_vector: List[float], num_candidates: int, limit: int,
) -> List[dict[str, Any]]:
    return [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit,
            },
        },
        {"$project": {"_id": 0, "text": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]


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
                "?retryWrites=true&w=majority",
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

    def forward(self, query_or_queries: str) -> dspy.Prediction:
        query_vector = self.embedder([query_or_queries])
        pipeline = build_vector_search_pipeline(
            index_name=self.index_name,
            query_vector=query_vector[0],
            num_candidates=self.k * 10,
            limit=self.k,
        )
        contents = self.client[self.db_name][self.collection_name].aggregate(pipeline)
        return dspy.Prediction(passages=list(contents))
