# HFClient vLLM

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Prerequisites

Follow these steps to set up the vLLM Server:

1. Build the server from source by following the instructions provided in the [Build from Source guide](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source).

2. Start the server by running the following command, and specify your desired model, host, and port using the appropriate arguments. The default server address is http://localhost:8000.

Example command:

```bash
   python -m vllm.entrypoints.openai.api_server --model mosaicml/mpt-7b --port 8000
```

This command will start the server and make it accessible at `http://localhost:8080`.

## Using the vLLM Client

After setting up the vLLM server and ensuring that it displays "Connected" when it's running, you can interact with it using the `HFClientVLLM`.

Initialize the `HFClientVLLM` within your program with the desired parameters. Here is an example call:

```python
   vllm_mpt = dspy.HFClientVLLM(model="mosaicml/mpt-7b", port=8000, url="http://localhost")
```
Customize the `model`, `port`, `url`, and `max_tokens` according to your requirements. The `model` parameter should be set to the specific Hugging Face model ID you wish to use.

Please refer to the [official vLLM repository](https://github.com/vllm-project/vllm) for more detailed information and documentation.

## Sending Requests via vLLM Client

1) _**Recommended**_ Configure default LM using `dspy.configure`.

This allows you to define programs in DSPy and simply call modules on your input fields, having DSPy internally call the prompt on the configured LM.

```python
dspy.configure(lm=vllm_mpt)

#Example DSPy CoT QA program
qa = dspy.ChainOfThought('question -> answer')

response = qa(question="What is the capital of Paris?") #Prompted to vllm_mpt
print(response.answer)
```

2) Generate responses using the client directly.

```python
response = vllm_mpt._generate(prompt='What is the capital of Paris?')
print(response)
```

## Under the Hood

### `__init__(self, model, port, url="http://localhost", **kwargs)`

The constructor initializes the `HFModel` base class to support the handling of prompting models, configuring the client for communicating with the hosted vLLM server to generate requests. This requires the following parameters:

- `model` (_str_): ID of model connected to the vLLM server.
- `port` (_int_): Port for communicating to the vLLM server. 
- `url` (_str_): Base URL of hosted vLLM server. This will often be `"http://localhost"`.
- `**kwargs`: Additional keyword arguments to configure the vLLM client.

Example of the vLLM constructor:

```python
class HFClientVLLM(HFModel):
    def __init__(self, model, port, url="http://localhost", **kwargs):
```

### `_generate(self, prompt, **kwargs) -> dict`

**Parameters:**
- `prompt` (_str_): Prompt to send to model hosted on vLLM server.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**
- `dict`: dictionary with `prompt` and list of response `choices`.

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response. 

After generation, the method parses the JSON response received from the server and retrieves the output through `json_response["choices"]` and stored as the `completions` list.

Lastly, the method constructs the response dictionary with two keys: the original request `prompt` and `choices`, a list of dictionaries representing generated completions with the key `text` holding the response's generated text.
