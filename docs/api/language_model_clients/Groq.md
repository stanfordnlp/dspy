---
sidebar_position: 9
---

# dspy.GROQ

### Usage

```python
lm = dspy.GroqLM(model='mixtral-8x7b-32768', api_key ="gsk_***" )
```

### Constructor

The constructor initializes the base class `LM` and verifies the provided arguments like the `api_key` for GROQ api retriver. The `kwargs` attribute is initialized with default values for relevant text generation parameters needed for communicating with the GROQ API, such as `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, and `n`.

```python
class GroqLM(LM):
    def __init__(
        self,
        api_key: str,
        model: str = "mixtral-8x7b-32768",
        **kwargs,
    ):
```



**Parameters:** 
- `api_key` str: API provider authentication token. Defaults to None.
- `model` str: model name. Defaults to "mixtral-8x7b-32768' options: ['llama2-70b-4096', 'gemma-7b-it']
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

####   `def __call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False, **kwargs, ) -> list[dict[str, Any]]:`

Retrieves completions from GROQ by calling `request`. 

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response.

After generation, the generated content look like `choice["message"]["content"]`. 

**Parameters:**
- `prompt` (_str_): Prompt to send to GROQ.
- `only_completed` (_bool_, _optional_): Flag to return only completed responses and ignore completion due to length. Defaults to True.
- `return_sorted` (_bool_, _optional_): Flag to sort the completion choices using the returned averaged log-probabilities. Defaults to False.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**
- `List[Dict[str, Any]]`: List of completion choices.
