# dspy.LlamaCpp

## Prerequisites

Install Llama Cpp Python by following the instructions provided in the [Llama Cpp Python repository](https://github.com/abetlen/llama-cpp-python).

```shell
pip install llama-cpp-python
```

alternatively, to install with CUDA support:

```shell
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```


### Initializing the Llama model

Initialize the model within your program with the desired parameters.

```python
from llama_cpp import Llama

llm = Llama(
      model_path="./sppo_finetuned_llama_3_8b.gguf",
      n_gpu_layers=-1,
      n_ctx=0,
      verbose=False
)
```


### Sending requests to the model

After initializing the Llama model, you can interact with it using the `LlamaCpp` client.

```python
import dspy

llamalm = dspy.LlamaCpp(model="llama", llama_model=llm,  model_type="chat", temperature=0.4)
```