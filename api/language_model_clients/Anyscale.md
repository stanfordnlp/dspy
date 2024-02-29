---
sidebar_position: 6
---

# dspy.Anyscale

### Usage

```python
lm = dspy.Anyscale(model="mistralai/Mistral-7B-Instruct-v0.1")
```

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` for using Anyscale API.
We expect the following environment variables to be set:
- `ANYSCALE_API_KEY`: API key for Together.
- `ANYSCALE_API_BASE`: API base URL for Together.


```python
class Anyscale(HFModel):
    def __init__(self, model, **kwargs):
```

**Parameters:**
- `model` (_str_): models hosted on Together.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.
