---
sidebar_position: 2
---

# retrieve.DatabricksRM

### Constructor

Initialize an instance of the `DatabricksRM` retriever class, which enables DSPy programs to query
[Databricks Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html#mosaic-ai-vector-search)
indexes for document retrieval.

```python
DatabricksRM(
    databricks_index_name: str,
    databricks_endpoint: Optional[str] = None,
    databricks_token: Optional[str] = None,
    columns: Optional[List[str]] = None,
    filters_json: Optional[str] = None,
    k: int = 3,
    docs_id_column_name: str = "id",
    text_column_name: str = "text",
)
```

**Parameters:**

- `databricks_index_name (str)`: The name of the Databricks Vector Search Index to query.
- `databricks_endpoint (Optional[str])`: The URL of the Databricks Workspace containing
  the Vector Search Index. Defaults to the value of the `DATABRICKS_HOST` environment variable.
  If unspecified, the Databricks SDK is used to identify the endpoint based on the current
  environment.
- `databricks_token (Optional[str])`: The Databricks Workspace authentication token to use
  when querying the Vector Search Index. Defaults to the value of the `DATABRICKS_TOKEN`
  environment variable. If unspecified, the Databricks SDK is used to identify the token based on
  the current environment.
- `columns (Optional[List[str]])`: Extra column names to include in response, in addition to the
  document id and text columns specified by `docs_id_column_name` and `text_column_name`.
- `filters_json (Optional[str])`: A JSON string specifying additional query filters.
  Example filters: `{"id <": 5}` selects records that have an `id` column value
  less than 5, and `{"id >=": 5, "id <": 10}` selects records that have an `id`
  column value greater than or equal to 5 and less than 10.
- `k (int)`: The number of documents to retrieve.
- `docs_id_column_name (str)`: The name of the column in the Databricks Vector Search Index
  containing document IDs.
- `text_column_name (str)`: The name of the column in the Databricks Vector Search Index
  containing document text to retrieve.

### Methods

#### `def forward(self, query: Union[str, List[float]], query_type: str = "ANN", filters_json: Optional[str] = None) -> dspy.Prediction:`

Retrieve documents from a Databricks Mosaic AI Vector Search Index that are relevant to the
specified query.

**Parameters:**

- `query (Union[str, List[float]])`: The query text or numeric query vector
  for which to retrieve relevant documents.
- `query_type (str)`: The type of search query to perform against the
  Databricks Vector Search Index. Must be either 'ANN' (approximate nearest neighbor) or 'HYBRID'
  (hybrid search).
- `filters_json (Optional[str])`: A JSON string specifying additional query filters.
  Example filters: `{"id <": 5}` selects records that have an `id` column value
  less than 5, and `{"id >=": 5, "id <": 10}` selects records that have an `id`
  column value greater than or equal to 5 and less than 10. If specified, this
  parameter overrides the `filters_json` parameter passed to the constructor.

**Returns:**

- `dspy.Prediction`: A `dotdict` containing retrieved documents. The schema is
  `{'docs': List[str], 'doc_ids': List[Any], extra_columns: List[Dict[str, Any]]}`.
  The `docs` entry contains the retrieved document content.

### Quickstart

To retrieve documents using Databricks Mosaic AI Vector Search, you must [create a
Databricks Mosaic AI Vector Search Index](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html)
first.

The following example code demonstrates how to set up a Databricks Mosaic AI
[Direct Access Vector Search Index](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)
and use the `DatabricksRM` DSPy retriever module to query the index. The example requires
the `databricks-vectorsearch` Python library to be installed.

```python
from databricks.vector_search.client import VectorSearchClient

# Create a Databricks Vector Search Endpoint
client = VectorSearchClient()
client.create_endpoint(
    name="your_vector_search_endpoint_name",
    endpoint_type="STANDARD"
)

# Create a Databricks Direct Access Vector Search Index
index = client.create_direct_access_index(
    endpoint_name="your_vector_search_endpoint_name",
    index_name="your_index_name",
    primary_key="id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector",
    schema={
      "id": "int",
      "field2": "str",
      "field3": "float",
      "text_vector": "array<float>"
    }
)

# Create a DatabricksRM retriever and retrieve the top-3 most relevant documents from the
# Databricks Direct Access Vector Search Index corresponding to an example query
retriever = DatabricksRM(
    databricks_index_name = "your_index_name",
    docs_id_column_name="id",
    text_column_name="field2",
    k=3
)
retrieved_results = DatabricksRM(query="Example query text", query_type="hybrid"))
```
