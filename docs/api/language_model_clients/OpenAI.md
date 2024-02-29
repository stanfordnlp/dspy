---
sidebar_position: 1
---

# dspy.OpenAI

### Usage

```python
lm = dspy.OpenAI(model='gpt-3.5-turbo')
```

### Constructor

The constructor initializes the base class `LM` and verifies the provided arguments like the `api_provider`, `api_key`, and `api_base` to set up OpenAI request retrieval. The `kwargs` attribute is initialized with default values for relevant text generation parameters needed for communicating with the GPT API, such as `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, and `n`.

```python
class OpenAI(LM):
    def __init__(
        self,
        model: str = "text-davinci-002",
        api_key: Optional[str] = None,
        api_provider: Literal["openai"] = "openai",
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
```



**Parameters:** 
- `api_key` (_Optional[str]_, _optional_): API provider authentication token. Defaults to None.
- `api_provider` (_Literal["openai"]_, _optional_): API provider to use. Defaults to "openai".
- `model_type` (_Literal["chat", "text"]_): Specified model type to use.
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

#### `__call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False, **kwargs) -> List[Dict[str, Any]]`

Retrieves completions from OpenAI by calling `request`. 

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response.

After generation, the completions are post-processed based on the `model_type` parameter. If the parameter is set to 'chat', the generated content look like `choice["message"]["content"]`. Otherwise, the generated text will be `choice["text"]`.

**Parameters:**
- `prompt` (_str_): Prompt to send to OpenAI.
- `only_completed` (_bool_, _optional_): Flag to return only completed responses and ignore completion due to length. Defaults to True.
- `return_sorted` (_bool_, _optional_): Flag to sort the completion choices using the returned averaged log-probabilities. Defaults to False.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**
- `List[Dict[str, Any]]`: List of completion choices.