# Using local models within DSPy

DSPy supports various methods including `built-in wrappers`, `server integration`, and `external package integration` for model loading. This documentation provides a concise introduction on how to load in models within DSPy extending these capabilities for your specific needs.

## Local Model Loaders

| Loaders | Jump To |
| --- | --- |
| HFModel | [HFModel Section](#hfmodel) |
| Cohere | [Cohere Section](#cohere) |
| TGI | [TGI Section](#tgi) |
| VLLM | [VLLM Section](#vllm) |


# HFModel

Initialize `HFModel` within your program with the desired model to load in. Here's an example call:

   ```python
   llama = dspy.HFModel(model = 'meta-llama/Llama-2-7b-hf')
   ```

# Text-Generation-Inference Server

## Prerequisites

- Docker must be installed on your system. If you don't have Docker installed, you can get it from [here](https://docs.docker.com/get-docker/).

## Setting up the Text-Generation-Inference Server

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

    docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data -e HUGGING_FACE_HUB_TOKEN={your_token} ghcr.io/huggingface/text-generation-inference:0.9.3 --model-id $model --num-shard $num_shard

## Sending requests to the server

After setting up the text-generation-inference server and ensuring that it displays "Connected" when it's running, you can interact with it using the `HFClientTGI`.

Initialize the `HFClientTGI` within your program with the desired parameters. Here is an example call:

   ```python
   lm = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
   ```

   Customize the `model`, `port`, and `url` according to your requirements. The `model` parameter should be set to the specific Hugging Face model ID you wish to use. 


### FAQs

1. If your model doesn't require any shards, you still need to set a value for `num_shard`, but you don't need to include the parameter `--num-shard` on the command line.

2. If your model runs into any "token exceeded" issues, you can set the following parameters on the command line to adjust the input length and token limit:
   - `--max-input-length`: Set the maximum allowed input length for the text.
   - `--max-total-tokens`: Set the maximum total tokens allowed for text generation.

Please refer to the [official Text-Generation-Inference repository](https://github.com/huggingface/text-generation-inference) for more detailed information and documentation.


# vLLM Server

## Setting up the vLLM Server

Follow these steps to set up the vLLM Server:

1. Build the server from source by following the instructions provided in the [Build from Source guide](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source).

2. Start the server by running the following command, and specify your desired model, host, and port using the appropriate arguments. The default server address is http://localhost:8000.

   Example command:
   ```
   python -m vllm.entrypoints.api_server --model mosaicml/mpt-7b --port 8000
   ```

This will launch the vLLM server.

## Sending requests to the vLLM server

After setting up the vLLM server and ensuring that it displays "Connected" when it's running, you can interact with it using the `HFClientVLLM`.

Initialize the `HFClientVLLM` within your program with the desired parameters. Here is an example call:

   ```python
   lm = dspy.HFClientVLLM(model="mosaicml/mpt-7b", port=8000, url="http://localhost")
   ```

   Customize the `model`, `port`, `url`, and `max_tokens` according to your requirements. The `model` parameter should be set to the specific Hugging Face model ID you wish to use.

Please refer to the [official vLLM repository](https://github.com/vllm-project/vllm) for more detailed information and documentation.

# MLC LLM

## Prerequisites

1. Install the required packages using the following commands:
   
   ```shell
   pip install --no-deps --pre --force-reinstall mlc-ai-nightly-cu118 mlc-chat-nightly-cu118 -f https://mlc.ai/wheels
   pip install transformers
   git lfs install
   ```
   
   Adjust the pip wheels according to your OS/platform by referring to the provided commands in [MLC packages](https://mlc.ai/package/).

## Running MLC Llama-2 models

1. Create a directory for prebuilt models:

   ```shell
   mkdir -p dist/prebuilt
   ```
   
2. Clone the necessary libraries from the repository:

   ```shell
   git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib
   cd dist/prebuilt
   ```
   
3. Choose a Llama-2 model from [MLC LLMs](https://huggingface.co/mlc-ai) and clone the model repository:

   ```shell
   git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
   ```

4. Initialize the `ChatModuleClient` within your program with the desired parameters. Here's an example call:

   ```python
   llama = dspy.ChatModuleClient(model='dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1', model_path='dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so')
   ```
Please refer to the [official MLC repository](https://github.com/mlc-ai/mlc-llm) for more detailed information and [documentation](https://mlc.ai/mlc-llm/docs/get_started/try_out.html).
