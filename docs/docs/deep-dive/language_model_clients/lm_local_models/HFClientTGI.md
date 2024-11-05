import AuthorDetails from '@site/src/components/AuthorDetails';

# HFClientTGI

## Prerequisites

Docker must be installed on your system. If you don't have Docker installed, you can get it from [here](https://docs.docker.com/get-docker/).

1. Clone the Text-Generation-Inference repository from GitHub by executing the following command:

   ```
   git clone https://github.com/huggingface/text-generation-inference.git
   ```

2. Change into the cloned repository directory:

   ```
   cd text-generation-inference
   ```

3. Execute the Docker command under the "Get Started" section to run the server:


   ```
   model=meta-llama/Llama-2-7b-hf # set to the specific Hugging Face model ID you wish to use.
   num_shard=2 # set to the number of shards you wish to use.
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.9 --model-id $model --num-shard $num_shard
   ```

   This command will start the server and make it accessible at `http://localhost:8080`.

If you want to connect to [Meta Llama 2 models](https://huggingface.co/meta-llama), make sure to use version 9.3 (or higher) of the docker image (ghcr.io/huggingface/text-generation-inference:0.9.3) and pass in your huggingface token as an environment variable.

```
   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data -e HUGGING_FACE_HUB_TOKEN={your_token} ghcr.io/huggingface/text-generation-inference:0.9.3 --model-id $model --num-shard $num_shard
```

## Using the TGI Client

After setting up the text-generation-inference server and ensuring that it displays "Connected" when it's running, you can interact with it using the `HFClientTGI`.

Initialize the `HFClientTGI` within your program with the desired parameters. Here is an example call:

```python
tgi_llama2 = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
```

Customize the `model`, `port`, and `url` according to your requirements. The `model` parameter should be set to the specific Hugging Face model ID you wish to use. 

## Sending Requests via TGI Client

1) _**Recommended**_ Configure default LM using `dspy.configure`.

This allows you to define programs in DSPy and simply call modules on your input fields, having DSPy internally call the prompt on the configured LM.

```python
dspy.configure(lm=tgi_llama2)

#Example DSPy CoT QA program
qa = dspy.ChainOfThought('question -> answer')

response = qa(question="What is the capital of Paris?") #Prompted to tgi_llama2
print(response.answer)
```

2) Generate responses using the client directly.

```python
response = tgi_llama2._generate(prompt='What is the capital of Paris?')
print(response)
```

## Under the Hood

### `__init__(self, model, port, url="http://future-hgx-1", http_request_kwargs=None, **kwargs)`

The constructor initializes the `HFModel` base class to support the handling of prompting HuggingFace models. It configures the client for communicating with the hosted TGI server to generate requests. This requires the following parameters:

- `model` (_str_): ID of Hugging Face model connected to the TGI server.
- `port` (_int_ or _list_): Port for communicating to the TGI server. This can be a single port number (`8080`) or a list of TGI ports (`[8080, 8081, 8082]`) to route the requests to.
- `url` (_str_): Base URL of hosted TGI server. This will often be `"http://localhost"`.
- `http_request_kwargs` (_dict_): Dictionary of additional keyword arguments to pass to the HTTP request function to the TGI server. This is `None` by default. 
- `**kwargs`: Additional keyword arguments to configure the TGI client.

Example of the TGI constructor:

```python
class HFClientTGI(HFModel):
    def __init__(self, model, port, url="http://future-hgx-1", http_request_kwargs=None, **kwargs):
```

### `_generate(self, prompt, **kwargs) -> dict`

**Parameters:**
- `prompt` (_str_): Prompt to send to model hosted on TGI server.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**
- `dict`: dictionary with `prompt` and list of response `choices`.

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response. 

After generation, the method parses the JSON response received from the server and retrieves the output through `json_response["generated_text"]`. This is then stored in the `completions` list.

If the JSON response includes the additional `details` argument and correspondingly, the `best_of_sequences` within `details`, this indicates multiple sequences were generated. This is also usually the case when `best_of > 1` in the initialized kwargs. Each of these sequences is accessed through `x["generated_text"]` and added to the `completions` list.

Lastly, the method constructs the response dictionary with two keys: the original request `prompt` and `choices`, a list of dictionaries representing generated completions with the key `text` holding the response's generated text.

## FAQs

1. If your model doesn't require any shards, you still need to set a value for `num_shard`, but you don't need to include the parameter `--num-shard` on the command line.

2. If your model runs into any "token exceeded" issues, you can set the following parameters on the command line to adjust the input length and token limit:
   - `--max-input-length`: Set the maximum allowed input length for the text.
   - `--max-total-tokens`: Set the maximum total tokens allowed for text generation.

Please refer to the [official Text-Generation-Inference repository](https://github.com/huggingface/text-generation-inference) for more detailed information and documentation.


***

<AuthorDetails name="Arnav Singhvi"/>