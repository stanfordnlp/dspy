---
sidebar_position: 7
---

# dspy.Together

### Usage

```python
lm = dspy.Together(model="mistralai/Mistral-7B-v0.1")
```

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` for using Together API.
We expect the following environment variables to be set:
- `TOGETHER_API_KEY`: API key for Together.
- `TOGETHER_API_BASE`: API base URL for Together.


```python
class Together(HFModel):
    def __init__(self, model, **kwargs):
```

**Parameters:**
- `model` (_str_): models hosted on Together.
- `stop` (_List[str]_, _optional_): List of stopping tokens to end generation.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.