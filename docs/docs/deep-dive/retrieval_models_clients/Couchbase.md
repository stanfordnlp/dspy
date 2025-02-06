# CouchbaseRM

A retrieval module that utilizes Couchbase's vector search capabilities to retrieve top passages for a given query.

## Prerequisites

```bash
pip install dspy-ai[couchbase]
```

## Setting up the CouchbaseRM Client

The constructor initializes an instance of the `CouchbaseRM` class and sets up parameters for connecting to Couchbase and performing vector searches.

### Parameters

- `cluster_connection_string` (str): Connection string for the Couchbase cluster.
- `bucket` (str): Name of the Couchbase bucket.
- `index_name` (str): Name of the vector search index.
- `k` (int, optional): Number of results to return. Defaults to 5.
- `text_field` (str, optional): Name of the field containing text content. Defaults to "text".
- `embedding_provider` (str, optional): Name of embedding provider. Defaults to "openai".
- `embedding_model` (str, optional): Name of embedding model. Defaults to "text-embedding-ada-002".
- `embedding_field` (str, optional): Name of field containing vector embeddings. Defaults to "embedding".
- `scope` (Optional[str]): Scope name (required for non-global index). Defaults to None.
- `collection` (Optional[str]): Collection name (required for KV get operation). Defaults to None.
- `cluster_options` (Optional[ClusterOptions]): Additional cluster connection options. Defaults to None.
- `is_global_index` (bool, optional): Whether to use global or scoped index. Defaults to False.
- `embedding_function` (Optional[Callable]): Optional custom embedding function. Defaults to None.
- `search_query` (Optional[SearchQuery]): Optional additional search query to combine with vector search. Defaults to None.
- `use_kv_get_text` (Optional[bool]): Whether to use KV get operation to fetch text field. Defaults to False.

Example of the CouchbaseRM constructor:

```python
from dspy.retrieve.couchbase_rm import CouchbaseRM

couchbase_rm = CouchbaseRM(
    cluster_connection_string="couchbase://localhost",
    bucket="my_bucket",
    index_name="vector_index",
    k=5,
    scope="my_scope",
    collection="my_collection",
    is_global_index=False,
    use_kv_get_text=True
)
```

## Under the Hood

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**

- `query_or_queries` (Union[str, List[str]]): The query or queries to search for.
- `k` (Optional[int]): Number of results to return. If not specified, uses the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

The method handles:

1. Converting queries to embeddings using the configured embedding provider
2. Creating and executing vector search requests
3. Optionally combining with traditional search queries
4. Retrieving text content either from search results or via KV operations

## Using CouchbaseRM

1. **Recommended**: Configure default RM using `dspy.configure`

    ```python
    import dspy
    from dspy.retrieve.couchbase_rm import CouchbaseRM

    couchbase_rm = CouchbaseRM(
        cluster_connection_string="couchbase://localhost",
        bucket="my_bucket",
        index_name="vector_index",
        k=5
    )

    dspy.settings.configure(rm=couchbase_rm)
    retrieve = dspy.Retrieve(k=5)
    retrieval_response = retrieve("What is quantum computing?").passages

    for result in retrieval_response:
        print("Text:", result, "\n")
    ```

2. Use the client directly:

    ```python
    from dspy.retrieve.couchbase_rm import CouchbaseRM

    couchbase_rm = CouchbaseRM(
        cluster_connection_string="couchbase://localhost",
        bucket="my_bucket",
        index_name="vector_index",
        k=5
    )

    retrieval_response = couchbase_rm("What is quantum computing?", k=5)
    for result in retrieval_response:
        print("Text:", result.long_text, "\n")
    ```

3. Example with custom embedding function:

    ```python
    def custom_embedder(queries):
        # Your custom embedding logic here
        return embeddings  # List[List[float]]

    couchbase_rm = CouchbaseRM(
        cluster_connection_string="couchbase://localhost",
        bucket="my_bucket",
        index_name="vector_index",
        embedding_function=custom_embedder,
        k=5
        )
    ```

## Note

- The CouchbaseRM client supports both global and scoped vector indexes in Couchbase.
- By default, it uses OpenAI for generating embeddings, but you can provide your own embedding function.
- The client can retrieve text content either from search results or via KV operations using `use_kv_get_text`.
- When using scoped indexes, both `scope` and `collection` parameters must be provided.
- The client automatically handles connection management and retries with backoff for embedding generation.

***
