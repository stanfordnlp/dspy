---
sidebar_position: 4
---

# dspy.HFClientTGI

### Usage

```python
lm = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
```

### Prerequisites

Refer to the [Text Generation-Inference Server](https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/local_models/HFClientTGI) section of the `Using Local Models` documentation.

### Constructor

The constructor initializes the `HFModel` base class and configures the client for communicating with the TGI server. It requires a `model` instance, communication `port` for the server, and the `url` for the server to host generate requests. Additional configuration can be provided via keyword arguments in `**kwargs`.

```python
class HFClientTGI(HFModel):
    def __init__(self, model, port, url="http://future-hgx-1", **kwargs):
```

**Parameters:**
- `model` (_HFModel_): Instance of Hugging Face model connected to the TGI server.
- `port` (_int_): Port for TGI server.
- `url` (_str_): Base URL where the TGI server is hosted. 
- `**kwargs`: Additional keyword arguments to configure the client.

### Methods

Refer to [`dspy.OpenAI`](https://dspy-docs.vercel.app/api/language_model_clients/OpenAI) documentation.