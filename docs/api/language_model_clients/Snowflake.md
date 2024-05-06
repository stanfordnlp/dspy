---
sidebar_position: 
---

# dspy.Snowflake

### Usage

```python
lm = dspy.Snowflake(model="mixtral-8x7b",credentials=connection_parameters)
```

### Constructor

The constructor inherits from the base class `LM` and verifies the `credentials` for using Snowflake API.

```python
class Snowflake(LM):
    def __init__(
        self, 
        model,
        credentials,
        **kwargs):
```

**Parameters:**
- `model` (_str_): model hosted by [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability).
- `credentials`  (_dict_): connection parameters required to initialize a [snowflake snowpark session](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/api/snowflake.snowpark.Session)

### Methods

Refer to [`dspy.Snowflake`](https://dspy-docs.vercel.app/api/language_model_clients/Snowflake) documentation.
