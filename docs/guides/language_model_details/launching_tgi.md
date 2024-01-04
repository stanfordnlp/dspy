## Launching a Text Generation Inference (TGI) Server

### Prerequisites

- Docker must be installed on your system. If you don't have Docker installed, you can get it from [here](https://docs.docker.com/get-docker/).

### Setting up the Text-Generation-Inference Server

1. Clone the Text-Generation-Inference repository from GitHub by executing the following command:

```bash
git clone https://github.com/huggingface/text-generation-inference.git
```

2. Change into the cloned repository directory:

```bash
cd text-generation-inference
```

3. Execute the Docker command under the "Get Started" section to run the server:

```bash
model=meta-llama/Llama-2-7b-hf # set to the specific Hugging Face model ID you wish to use.
num_shard=1 # set to the number of shards you wish to use.
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --num-shard $num_shard
```

This command will start the server and make it accessible at `http://localhost:8080`.

If you want to connect to [Meta Llama 2 models](https://huggingface.co/meta-llama), make sure to use version 9.3 (or higher) of the docker image (ghcr.io/huggingface/text-generation-inference:0.9.3) and pass in your huggingface token as an environment variable.

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data -e HUGGING_FACE_HUB_TOKEN={your_token} ghcr.io/huggingface/text-generation-inference:latest --model-id $model --num-shard $num_shard
```

### Sending requests to the server

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

Please refer to the [official TGI repository](https://github.com/huggingface/text-generation-inference) for detailed docs.
