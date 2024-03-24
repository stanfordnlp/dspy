---
sidebar_position: 9
---

# dsp.Mistral

### Usage

```python
lm = dsp.Mistral(model='mistral-medium-latest', api_key="your-mistralai-api-key")
```

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` provided or defined through the `MISTRAL_API_KEY` environment variable.

```python
class Mistral(LM):
    def __init__(
        self,
        model: str = "mistral-medium-latest",
        api_key: Optional[str] = None,
        **kwargs,
    ):
```

**Parameters:**
- `model` (_str_): Mistral AI pretrained models. Defaults to `mistral-medium-latest`.
- `api_key` (_Optional[str]_, _optional_): API provider from Mistral AI. Defaults to None.
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

Refer to [`dspy.Mistral`](#) documentation.
