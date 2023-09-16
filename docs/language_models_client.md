# dspy.LM Documentation

This documentation provides an overview of the DSPy Language Model Clients.

## dspy.OpenAI

### Quickstart

```python
import dspy

lm = dspy.OpenAI(model='gpt-3.5-turbo')

prompt = "Translate the following English text to Spanish: 'Hi, how are you?'"
completions = lm(prompt, n=5, return_sorted=True)
for i, completion in enumerate(completions):
    print(f"Completion {i+1}: {completion}")
```



### Constructor

The constructor initializes the base class `LM` and verifies the provided arguments like the `api_provider`, `api_key`, and `api_base` to set up OpenAI request retrieval. The `kwargs` attribute is initialized with default values for relevant text generation parameters needed for communicating with the GPT API, such as `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, and `n`.

```python
class GPT3(LM):
    def __init__(
        self,
        model: str = "text-davinci-002",
        api_key: Optional[str] = None,
        api_provider: Literal["openai", "azure"] = "openai",
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
```

**Arguments:**
- `model` (str): OpenAI or Azure supported LLM model to use. Defaults to OpenAI's `"text-davinci-002"`.
- `api_key` (Optional[str], optional): API provider authentication token. Defaults to None.
- `api_provider` (Literal["openai", "azure"], optional): API provider to use. Defaults to "openai".
- `model_type` (Literal["chat", "text"]): Specified model type to use.
- `**kwargs`: Additional language model arguments to pass to the API provider.

### Methods

#### `__call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False, **kwargs) -> List[Dict[str, Any]]`

Retrieves completions from GPT-3 by calling `request`. 

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response.

After generation, the completions are post-processed based on the `model_type` parameter. If the parameter is set to 'chat', the generated content look like `choice["message"]["content"]`. Otherwise, the generated text will be `choice["text"]`.

**Arguments:**
- `prompt` (str): Prompt to send to GPT-3.
- `only_completed` (bool, optional): Flag to return only completed responses and ignore completion due to length. Defaults to True.
- `return_sorted` (bool, optional): Flag to sort the completion choices using the returned averaged log-probabilities. Defaults to False.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**
- `List[Dict[str, Any]]`: List of completion choices.

## dspy.Cohere

### Constructor

The constructor initializes the base class `LM` and verifies the `api_key` to set up OpenAI request retrieval.

```python
class Cohere(LM):
    def __init__(
        self,
        model: str = "command-xlarge-nightly",
        api_key: Optional[str] = None,
        stop_sequences: List[str] = [],
    ):
```

**Arguments:**
- `model` (str): Cohere pretrained models. Defaults to `command-xlarge-nightly`.
- `api_key` (Optional[str], optional): API provider from Cohere. Defaults to None.
- `stop_sequences` (List[str], optional): List of stopping tokens to end generation.

### Methods

Refer to `dspy.OpenAI` documentation.

### Examples

Refer to `dspy.OpenAI` documentation. Replace with

```python
lm = dsp.Cohere(model='command-xlarge-nightly')
```

## dspy.TGI

### Prerequisites

Refer to the [Text Generation-Inference Server](https://github.com/stanfordnlp/dspy/blob/local_models_docs/docs/using_local_models.md#text-generation-inference-server) section of the `Using Local Models` documentation.

### Constructor

The constructor initializes the `HFModel` base class and configures the client for communicating with the TGI server. It requires a `model` instance, communication `port` for the server, and the `url` for the server to host generate requests. Additional configuration can be provided via keyword arguments in `**kwargs`.

```python
class HFClientTGI(HFModel):
    def __init__(self, model, port, url="http://future-hgx-1", **kwargs):
```

**Arguments:**
- `model` (HFModel): Instance of Hugging Face model connected to the TGI server.
- `port` (int): Port for TGI server.
- `url` (str): Base URL where the TGI server is hosted. 
- `**kwargs`: Additional keyword arguments to configure the client.

### Methods

Refer to `dspy.OpenAI` documentation.

### Examples

Refer to `dspy.OpenAI` documentation. Replace with

```python
lm = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
```

## dspy.VLLM

### Prerequisites

Refer to the [vLLM Server](https://github.com/stanfordnlp/dspy/blob/local_models_docs/docs/using_local_models.md#vllm-server) section of the `Using Local Models` documentation.

### Constructor

Refer to `dspy.TGI` documentation. Replace with `HFClientVLLM`.

### Methods

Refer to `dspy.OpenAI` documentation.

### Examples

Refer to `dspy.OpenAI` documentation. Replace with

```python
lm = dspy.HFClientVLLM(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
```