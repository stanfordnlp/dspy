---
sidebar_position: 3
---

# dsp.Cohere

### Usage

```python
lm = dsp.Cohere(model='command-nightly')
```

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` to set up Cohere request retrieval.

```python
class Cohere(LM):
    def __init__(
        self,
        model: str = "command-nightly",
        api_key: Optional[str] = None,
        stop_sequences: List[str] = [],
    ):
```

**Parameters:**
- `model` (_str_): Cohere pretrained models. Defaults to `command-nightly`.
- `api_key` (_Optional[str]_, _optional_): API provider from Cohere. Defaults to None.
- `stop_sequences` (_List[str]_, _optional_): List of stopping tokens to end generation.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.
