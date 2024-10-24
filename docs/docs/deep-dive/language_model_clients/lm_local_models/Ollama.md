# OllamaLocal

!!! note
    Adapted from documentation provided by https://github.com/insop

Ollama is a good software tool that allows you to run LLMs locally, such as Mistral, Llama2, and Phi.
The following are the instructions to install and run Ollama.

### Prerequisites

Install Ollama by following the instructions from this page:

- https://ollama.ai

Download model: `ollama pull`

Download a model by running the `ollama pull` command. You can download Mistral, Llama2, and Phi.

```bash
# download mistral
ollama pull mistral
```

Here is the list of other models you can download:
- https://ollama.ai/library

### Running Ollama model

Run model: `ollama run`

You need to start the model server with the `ollama run` command.

```bash
# run mistral
ollama run mistral
```

### Sending requests to the server

Here is the code to load a model through Ollama:

```python
lm = dspy.OllamaLocal(model='mistral')
```
