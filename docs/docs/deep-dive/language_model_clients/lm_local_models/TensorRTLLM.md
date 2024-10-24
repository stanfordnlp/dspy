# TensorRTModel

TensorRT LLM by Nvidia happens to be one of the most optimized inference engines to run open-source Large Language Models locally or in production.

### Prerequisites

Install TensorRT LLM by the following instructions [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html). You need to install `dspy` inside the same Docker environment in which `tensorrt` is installed.

In order to use this module, you should have the model weights file in engine format. To understand how we convert weights in torch (from HuggingFace models) to TensorRT engine format, you can check out [this documentation](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#build-tensorrt-engines).

### Running TensorRT model inside dspy

```python
from dspy import TensorRTModel

engine_dir = "<your-path-to-engine-dir>"
model_name_or_path = "<hf-model-id-or-path-to-tokenizer>"

model = TensorRTModel(engine_dir=engine_dir, model_name_or_path=model_name_or_path)
```

You can perform more customization on model loading based on the following example. Below is a list of optional parameters that are supported while initializing the `dspy` TensorRT model.

- **use_py_session** (`bool`, optional): Whether to use a Python session or not. Defaults to `False`.
- **lora_dir** (`str`): The directory of LoRA adapter weights.
- **lora_task_uids** (`List[str]`): List of LoRA task UIDs; use `-1` to disable the LoRA module.
- **lora_ckpt_source** (`str`): The source of the LoRA checkpoint.

If `use_py_session` is set to `False`, the following kwargs are supported (This runs in C++ runtime):

- **max_batch_size** (`int`, optional): The maximum batch size. Defaults to `1`.
- **max_input_len** (`int`, optional): The maximum input context length. Defaults to `1024`.
- **max_output_len** (`int`, optional): The maximum output context length. Defaults to `1024`.
- **max_beam_width** (`int`, optional): The maximum beam width, similar to `n` in OpenAI API. Defaults to `1`.
- **max_attention_window_size** (`int`, optional): The attention window size that controls the sliding window attention / cyclic KV cache behavior. Defaults to `None`.
- **sink_token_length** (`int`, optional): The sink token length. Defaults to `1`.

> Please note that you need to complete the build processes properly before applying these customizations, because a lot of customization depends on how the model engine was built. You can learn more [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#build-tensorrt-engines).

Now to run the model, we need to add the following code:

```python
response = model("hello")
```

This gives this result:

```
["nobody is perfect, and we all have our own unique struggles and challenges. But what sets us apart is how we respond to those challenges. Do we let them define us, or do we use them as opportunities to grow and learn?\nI know that I have my own personal struggles, and I'm sure you do too. But I also know that we are capable of overcoming them, and becoming the best versions of ourselves. So let's embrace our imperfections, and use them to fuel our growth and success.\nRemember, nobody is perfect, but everybody has the potential to be amazing. So let's go out there and make it happen!"]
```

You can also invoke chat mode by just changing the prompt to chat format like this:

```python
prompt = [{"role":"user", "content":"hello"}]
response = model(prompt)

print(response)
```

Output:

```
[" Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?"]
```

Here are some optional parameters that are supported while doing generation:

- **max_new_tokens** (`int`): The maximum number of tokens to output. Defaults to `1024`.
- **max_attention_window_size** (`int`): Defaults to `None`.
- **sink_token_length** (`int`): Defaults to `None`.
- **end_id** (`int`): The end of sequence ID of the tokenizer, defaults to the tokenizer's default end ID.
- **pad_id** (`int`): The pad sequence ID of the tokenizer, defaults to the tokenizer's default end ID.
- **temperature** (`float`): The temperature to control probabilistic behavior in generation. Defaults to `1.0`.
- **top_k** (`int`): Defaults to `1`.
- **top_p** (`float`): Defaults to `1`.
- **num_beams** (`int`): The number of responses to generate. Defaults to `1`.
- **length_penalty** (`float`): Defaults to `1.0`.
- **repetition_penalty** (`float`): Defaults to `1.0`.
- **presence_penalty** (`float`): Defaults to `0.0`.
- **frequency_penalty** (`float`): Defaults to `0.0`.
- **early_stopping** (`int`): Use this only when `num_beams` > 1. Defaults to `1`.
