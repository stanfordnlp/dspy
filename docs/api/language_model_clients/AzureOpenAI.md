---
sidebar_position: 2
---

# dspy.AzureOpenAI

### Usage

```python
lm = dspy.AzureOpenAI(api_base='...', api_version='2023-12-01-preview', model='gpt-3.5-turbo')
```

### Constructor

The constructor initializes the base class `LM` and verifies the provided arguments like the `api_provider`, `api_key`, and `api_base` to set up OpenAI request retrieval through Azure. The `kwargs` attribute is initialized with default values for relevant text generation parameters needed for communicating with the GPT API, such as `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, and `n`.

Azure requires that the deployment id of the Azure deployment to be also provided using the argument `deployment_id`.

```python
class AzureOpenAI(LM):
    def __init__(
        self,
        api_base: str,
        api_version: str,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
```

**Parameters:**

- `api_base` (str): Azure Base URL.
- `api_version` (str): Version identifier for Azure OpenAI API.
- `api_key` (_Optional[str]_, _optional_): API provider authentication token. Retrieves from `AZURE_OPENAI_KEY` environment variable if None.
- `model_type` (_Literal["chat", "text"]_): Specified model type to use, defaults to 'chat'.
- `azure_ad_token_provider` (_Optional[AzureADTokenProvider]_, _optional_): Pass the bearer token provider created by _get_bearer_token_provider()_ when using DefaultAzureCredential, alternative to api token.
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

#### `__call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False, **kwargs) -> List[Dict[str, Any]]`

Retrieves completions from Azure OpenAI Endpoints by calling `request`.

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response.

After generation, the completions are post-processed based on the `model_type` parameter. If the parameter is set to 'chat', the generated content look like `choice["message"]["content"]`. Otherwise, the generated text will be `choice["text"]`.

**Parameters:**

- `prompt` (_str_): Prompt to send to Azure OpenAI.
- `only_completed` (_bool_, _optional_): Flag to return only completed responses and ignore completion due to length. Defaults to True.
- `return_sorted` (_bool_, _optional_): Flag to sort the completion choices using the returned averaged log-probabilities. Defaults to False.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**

- `List[Dict[str, Any]]`: List of completion choices.
