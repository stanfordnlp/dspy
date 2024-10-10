---
sidebar_position: 12
---

# dspy.Snowflake

### Usage

```python
import dspy
import os
from snowflake.snowpark import Session

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

# Initialize Snowflake Cortex LM
lm = dspy.Snowflake(session=snowpark,model="mixtral-8x7b")
```

### Constructor

The constructor inherits from the base class `LM` and verifies the `session` for using Snowflake API.

```python
class Snowflake(LM):
    def __init__(
        self,
        session,
        model,
        **kwargs):
```

**Parameters:**

- `session` (_object_): Snowflake connection object enabled by the [snowflake snowpark session](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.Session)
- `model` (_str_): model hosted by [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability).
  snowpark/reference/python/latest/api/snowflake.snowpark.Session)

### Methods

Refer to [`dspy.Snowflake`](https://dspy-docs.vercel.app/api/language_model_clients/Snowflake) documentation.
