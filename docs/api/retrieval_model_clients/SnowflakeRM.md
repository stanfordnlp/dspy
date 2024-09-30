---
sidebar_position: 10
---

# retrieve.SnowflakeRM

### Constructor

Initialize an instance of the `SnowflakeRM` class, which enables user to leverage the Cortex Search service for hybrid retrieval. Before using this, ensure the Cortex Search service is configured as outlined in the documentation [here](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview#overview)

```python
SnowflakeRM(
     snowflake_session: object,
     cortex_search_service: str,
     snowflake_database: str,
     snowflake_schema: dict,
     auto_filter:bool,
     k: int = 3,
)
```

**Parameters:**

- `snowflake_session (str)`: Snowflake Snowpark session for connecting to Snowflake.
- `cortex_search_service (str)`: The name of the Cortex Search service to be used.
- `snowflake_database (str)`: The name of the Snowflake database to be used with the Cortex Search service.
- `snowflake_schema (str)`: The name of the Snowflake schema to be used with the Cortex Search service.
- `auto_filter (bool)`: Auto-generate metadata filter and push it down to Cortex Search service prior to retrieving results.
- `k (int, optional)`: The number of top passages to retrieve. Defaults to 3.

### Methods

#### `def forward(self,query_or_queries: Union[str, list[str]],response_columns:list[str],filters:dict = None, k: Optional[int] = None)-> dspy.Prediction:`

Query the Cortex Search service to retrieve the top k relevant results given a query.

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `retrieval_columns` (str)`: A list of columns to return for each relevant result in the response.
- `search_filter` (_Optional[dict]_): Optional filter object used for filtering results based on data in the ATTRIBUTES columns. See [Filter syntax](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/query-cortex-search-service#filter-syntax)
- `k` (_Optional[int]_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"long_text": str}]`

### Quickstart

To support passage retrieval from a Snowflake table with this integration, a [Cortex Search](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview) endpoint must first be configured.

```python
from dspy.retrieve.snowflake_rm import SnowflakeRM
from snowflake.snowpark import Session
import os

connection_parameters = {

    "account": os.getenv('SNOWFLAKE_ACCOUNT'),
    "user": os.getenv('SNOWFLAKE_USER'),
    "password": os.getenv('SNOWFLAKE_PASSWORD'),
    "role": os.getenv('SNOWFLAKE_ROLE'),
    "warehouse": os.getenv('SNOWFLAKE_WAREHOUSE'),
    "database": os.getenv('SNOWFLAKE_DATABASE'),
    "schema": os.getenv('SNOWFLAKE_SCHEMA')}

# Establish connection to Snowflake
snowpark = Session.builder.configs(connection_parameters).create()

snowflake_retriever = SnowflakeRM(snowflake_session=snowpark,
    cortex_search_service="<YOUR_CORTEX_SERACH_SERVICE_NAME>",
    snowflake_database="<YOUR_SNOWFLAKE_DATABASE_NAME>",
    snowflake_schema="<YOUR_SNOWFLAKE_SCHEMA_NAME>",
    auto_filter=True,
    k = 5)

results = snowflake_retriever("Explore the meaning of life",
    response_columns=["<NAME_OF_INDEXED_COLUMN>","<NAME_OF_ATTRIBUTE_COLUMN"])

for result in results:
    print("Document:", result.long_text, "\n")
```
