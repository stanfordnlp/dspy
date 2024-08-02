---
sidebar_position: 1
---

# dspy.MultiOpenAI

### Usage

```python
openrouter = dspy.MultiOpenAI(model='openai/gpt-4o-mini', 
                              api_key='xxxx',
                              api_provider='openrouter',
                              api_base='https://openrouter.ai/api/v1/',
                              model_type='chat',
                              )

siliconflow = dspy.MultiOpenAI(model='zhipuai/glm4-9B-chat', 
                               api_key='xxxx',
                               api_provider='siliconflow',
                               api_base='https://api.siliconflow.cn/v1/',
                               model_type='chat',
                               )
```

### Constructor

The constructor initializes the base class `LM` and verifies the provided arguments like the `model`, `api_provider`, `api_key`, and `api_base` to set up OpenAI request retrieval. The `kwargs` attribute is initialized with default values for relevant text generation parameters needed for communicating with the GPT API, such as `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, and `n`.

```python
class MultiOpenAI(LM):
    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        api_provider: str,
        api_base: str,
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
```



**Parameters:** 
- `model` (str): LLM model to use.
- `api_key` (Optional[str]): API provider Authentication token.
- `api_provider` (str): The API provider to use.
- `model_type` (_Literal["chat", "text"]_): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "chat".
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