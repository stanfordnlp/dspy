## Launching a vLLM Server

### Setting up the vLLM Server

Follow these steps to set up the vLLM Server:

1. Build the server from source by following the instructions provided in the [Build from Source guide](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source).

2. Start the server by running the following command, and specify your desired model, host, and port using the appropriate arguments. The default server address is http://localhost:8000.

Example command:

```bash
   python -m vllm.entrypoints.api_server --model mosaicml/mpt-7b --port 8000
```

This will launch the vLLM server.

### Sending requests to the server

After setting up the vLLM server and ensuring that it displays "Connected" when it's running, you can interact with it using the `HFClientVLLM`.

Initialize the `HFClientVLLM` within your program with the desired parameters. Here is an example call:

```python
   lm = dspy.HFClientVLLM(model="mosaicml/mpt-7b", port=8000, url="http://localhost")
```

Customize the `model`, `port`, `url`, and `max_tokens` according to your requirements. The `model` parameter should be set to the specific Hugging Face model ID you wish to use.

Please refer to the [official vLLM repository](https://github.com/vllm-project/vllm) for more detailed information and documentation.
