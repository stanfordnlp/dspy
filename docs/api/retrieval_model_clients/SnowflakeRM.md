---
sidebar_position: 
---

# retrieve.SnowflakeRM

### Constructor

Initialize an instance of the `SnowflakeRM` class, with the option to use `e5-base-v2` embeddings or any Snowflake Cortex supported embeddings model.

```python
SnowflakeRM(
     snowflake_table_name: str,
     snowflake_credentials: dict,
     k: int = 3,
     embeddings_field: str,
     embeddings_text_field:str,
     embeddings_model: str = "e5-base-v2",
)
```

**Parameters:**
- `snowflake_table_name (str)`: The name of the Snowflake table containing embeddings.
- `snowflake_credentials (dict)`: The connection parameters needed to initialize a Snowflake Snowpark Session.
- `k (int, optional)`: The number of top passages to retrieve. Defaults to 3.
- `embeddings_field (str)`: The name of the column in the Snowflake table containing the embeddings.
- `embeddings_text_field (str)`: The name of the column in the Snowflake table containing the passages.
- `embeddings_model (str)`: The model to be used to convert text to embeddings

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

Search the Snowflake table for the top `k` passages matching the given query or queries, using embeddings generated via the default `e5-base-v2` model or the specified `embedding_model`.

**Parameters:**
- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**
- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"id": str, "score": float, "long_text": str, "metadatas": dict }]`

### Quickstart

To support passage retrieval, it assumes that a Snowflake table has been created and populated with the passages in a column `embeddings_text_field` and the embeddings in another column `embeddings_field`

SnowflakeRM uses `e5-base-v2` embeddings model by default or any Snowflake Cortex supported embeddings model.

#### Default OpenAI Embeddings

```python
from dspy.retrieve.snowflake_rm import SnowflakeRM
import os

connection_parameters = {
    
    "account": os.getenv('SNOWFLAKE_ACCOUNT'),
    "user": os.getenv('SNOWFLAKE_USER'),
    "password": os.getenv('SNOWFLAKE_PASSWORD'),
    "role": os.getenv('SNOWFLAKE_ROLE'),
    "warehouse": os.getenv('SNOWFLAKE_WAREHOUSE'),
    "database": os.getenv('SNOWFLAKE_DATABASE'),
    "schema": os.getenv('SNOWFLAKE_SCHEMA')}  

retriever_model = SnowflakeRM(
    snowflake_table_name="<YOUR_SNOWFLAKE_TABLE_NAME>",
    snowflake_credentials=connection_parameters,
    embeddings_field="<YOUR_EMBEDDINGS_COLUMN_NAME>",
    embeddings_text_field= "<YOUR_PASSAGE_COLUMN_NAME>"
    )

results = retriever_model("Explore the meaning of life", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```

