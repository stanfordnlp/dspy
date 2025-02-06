import os
from datetime import timedelta
from typing import Any, Callable, List, Optional, Union, Dict
from couchbase.result import SearchResult
from dspy.dsp.utils.utils import dotdict

import backoff
from openai import (
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)

from dspy import Prediction, Retrieve
from dspy.dsp.utils.settings import settings

try:
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions, SearchOptions
    from couchbase.vector_search import VectorQuery, VectorSearch
    from couchbase import search
    from couchbase.search import SearchQuery
except ImportError:
    raise ImportError(
        "Please install the couchbase package by running `pip install dspy-ai[couchbase]`",
    )


class Embedder:
    """OpenAI embeddings provider."""
    
    def __init__(self, provider: str, model: str) -> None:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Environment variable OPENAI_API_KEY must be set")
            self.client = OpenAI()
            self.model = model

    @backoff.on_exception(
        backoff.expo,
        (APITimeoutError, InternalServerError, RateLimitError, UnprocessableEntityError),
        max_time=settings.backoff_time,
    )
    def __call__(self, queries: List[str]) -> List[List[float]]:
        embedding = self.client.embeddings.create(input=queries, model=self.model)
        return [result.embedding for result in embedding.data]


class CouchbaseRM(Retrieve):
    """Couchbase vector search retriever with support for global and scoped indexes."""

    def __init__(
        self,
        cluster_connection_string: str,
        bucket: str,
        index_name: str,
        k: int = 5,
        text_field: str = "text",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
        embedding_field: str = "embedding",
        scope: Optional[str] = None,
        collection: Optional[str] = None,
        cluster_options: Optional[ClusterOptions] = None,
        is_global_index: bool = False,
        embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
        search_query: Optional[SearchQuery] = None,
        use_kv_get_text: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Initialize CouchbaseRM.
        
        Args:
            cluster_connection_string: Connection string for Couchbase cluster
            bucket: Bucket name
            index_name: Name of the vector search index
            k: Number of results to return
            embedding_provider: Name of embedding provider (e.g. "openai")
            embedding_model: Name of embedding model
            embedding_field: Name of field containing vector embeddings
            scope: Scope name (required for non-global index)
            collection: Collection name (required for kv get operation)
            cluster_options: Additional cluster connection options
            is_global_index: Whether to use global or scoped index
            embedding_function: Optional custom embedding function
            search_query: Optional additional search query to combine with vector search
            use_kv_get_text: Whether to use KV get operation to fetch text field instead of search response
            
        Raises:
            ValueError: If scope/collection configuration is invalid
            ConnectionError: If connection to Couchbase cluster fails
        """
        super().__init__(k=k)
        
        # Store configuration
        self.cluster_connection_string = cluster_connection_string
        self.bucket_name = bucket
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.is_global_index = is_global_index
        self.use_kv_get_text = use_kv_get_text
        self.search_query = search_query
        self.scope_name = scope
        self.collection_name = collection

        # Initialize as None - will be set up lazily
        self._bucket = None
        self._scope = None
        self._collection = None
        
        # Initialize cluster connection and validate configuration
        self._initialize_cluster(cluster_options)
        
        # Set up embedder
        self.embedder = embedding_function or Embedder(
            provider=embedding_provider,
            model=embedding_model
        )

    @property
    def bucket(self):
        """Lazily initialized bucket object singleton."""
        if self._bucket is None:
            self._bucket = self.cluster.bucket(self.bucket_name)
            print("bucket", self._bucket)
        return self._bucket

    @property
    def scope(self):
        """Lazily initialized scope object singleton."""
        if self._scope is None and self.scope_name:
            self._scope = self.bucket.scope(self.scope_name)
        return self._scope
    
    @property
    def collection(self):
        """Lazily initialized collection object singleton."""
        if self._collection is None and self.collection_name:
            self._collection = self.scope.collection(self.collection_name)
        return self._collection

    def _initialize_cluster(self, cluster_options: Optional[ClusterOptions]) -> None:
        """Initialize and validate cluster connection."""
        try:
            self.cluster = Cluster(
                self.cluster_connection_string,
                cluster_options
            )
            self.cluster.wait_until_ready(timeout=timedelta(seconds=60))

            _ = self.bucket
            if not self.is_global_index and self.scope_name:
                _ = self.scope
                if self.collection_name:
                    _ = self.collection
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Couchbase cluster: {str(e)}") from e


    def _create_vector_request(self, vector: List[float]) -> search.SearchRequest:
        """Create a vector search request."""
        vector_query = VectorQuery(
            field_name=self.embedding_field,
            vector=vector,
            num_candidates=self.k
        )
        
        vector_search = VectorSearch.from_vector_query(vector_query)
        request = search.SearchRequest.create(vector_search)
        
        if self.search_query:
            request.with_search_query(self.search_query)
            
        return request

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> Prediction:
        """Execute vector search and return relevant passages.
        
        Args:
            query: Search query string
            k: Number of results to return (default: 3)
            **kwargs: Additional arguments
            
        Returns:
            Prediction containing matching passages
            
        Raises:
            RuntimeError: If search execution fails
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries
        query_vectors = self.embedder([queries])
        contents = []
        
        options = SearchOptions(
            fields=["*"],
            limit=k if k else self.k,
            timeout=timedelta(seconds=10)
        )

        for vector in query_vectors:
            request = self._create_vector_request(vector)
            try:
                response = (
                    self.cluster.search(self.index_name, request, options)
                    if self.is_global_index
                    else self.scope.search(self.index_name, request, options)
                )
                if not self.use_kv_get_text:
                    print("response", response.rows(), self.is_global_index)
                    contents.extend([dotdict({"long_text": hit["fields"][self.text_field]}) for hit in response.rows()])
                    #print("contents", contents)
                else:
                    results = self.__get_docs_from_kv(response)
                    contents.extend([dotdict({"long_text": doc[self.text_field] }) for doc in results])
            except Exception as e:
                search_type = "global" if self.is_global_index else "scoped"
                raise RuntimeError(f"Failed to execute {search_type} vector search: {str(e)}") from e

        return contents

    def __get_docs_from_kv(self, response: SearchResult) -> List[Dict]:
        documents: List[Dict] = []
        ids: List[str] = []
        scores: List[float] = []
        for doc in response.rows():
            ids.append(doc.id)
            scores.append(doc.score)
        kv_response = self.collection.get_multi(keys=ids)
        if not kv_response.all_ok and kv_response.exceptions:
            errors = []
            for id, ex in kv_response.exceptions.items():
                errors.append({"id": id, "exception": ex})
            if len(errors) > 0:
                msg = f"Failed to write documents to couchbase. Errors:\n{errors}"
                raise RuntimeError(msg)
        for i, id in enumerate(ids):
            get_result = kv_response.results.get(id)
            if get_result is not None and get_result.success:
                value = get_result.value
                value["id"] = id
                value["score"] = scores[i]
            documents.append(value)
            
        return documents
