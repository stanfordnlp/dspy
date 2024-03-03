---
sidebar_position: 8
---

# dspy.Databricks

### Usage
```python
lm = dspy.Databricks(model="databricks-mpt-30b-instruct")
```

### Constructor

The constructor inherits from the `GPT3` class and verifies the Databricks authentication credentials for using Databricks Model Serving API through the OpenAI SDK.
We expect the following environment variables to be set:
- `openai.api_key`: Databricks API key.
- `openai.base_url`: Databricks Model Endpoint url

The `kwargs` attribute is initialized with default values for relevant text generation parameters needed for communicating with the Databricks OpenAI SDK, such as `temperature`, `max_tokens`, `top_p`, and `n`. However, it removes the `frequency_penalty` and `presence_penalty` arguments as these are not currently supported by the Databricks API.

```python
class Databricks(GPT3):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
```

**Parameters:**
- `model` (_str_): models hosted on Databricks.
- `stop` (_List[str]_, _optional_): List of stopping tokens to end generation.
- `api_key` (_Optional[str]_): Databricks API key. Defaults to None
- `api_base` (_Optional[str]_): Databricks Model Endpoint url Defaults to None.
- `model_type` (_Literal["chat", "text", "embeddings"]_): Specified model type to use.
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.