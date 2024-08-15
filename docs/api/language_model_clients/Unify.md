---
sidebar_position: 15
---

# dspy.Unify

### Usage

Obtain a You.com API key from https://unify.ai/.

```python
lm = dspy.Unify(model="gpt-4o@openai", api_key="your_unify_api_key")
```

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` for using Unify AI API.

```python
class Unify(LM):
    def __init__(
        self,
        model: str = "gpt-4o@openai",
        api_key: Optional[str] = None,
        **kwargs,
    ):
```

**Parameters:**

- `model` (_str_): Endpoint routed by Unify. Defaults to `gpt-4o@openai`.
- `api_key` (_Optional[str]_, _optional_): API key for Unify. Defaults to None.
- `\*\*kwargs: Additional language model arguments to pass to the API provider.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.
