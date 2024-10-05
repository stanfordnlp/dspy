---
sidebar_position: 2
---

# Language Models

The most powerful features in DSPy revolve around algorithmically optimizing the prompts (or weights) of LMs, especially when you're building programs that use the LMs within a pipeline.

Let's first make sure you can set up your language model. DSPy support clients for many remote and local LMs.

## Using `dspy.LM`

:::warning
Earlier versions of DSPy involved tons of clients for different LM providers, e.g. `dspy.OpenAI`, `dspy.GoogleVertexAI`, and `dspy.HFClientTGI`, etc. These are now deprecated and will be removed in DSPy 2.6.

Instead, use `dspy.LM` to access any LM endpoint for local and remote models. This relies on [LiteLLM](https://github.com/BerriAI/litellm) to translate the different client APIs into an OpenAI-compatible interface.

Any [provider supported in LiteLLM](https://docs.litellm.ai/docs/providers) should work with `dspy.LM`.
:::

### Setting up the LM client

In DSPy 2.5, we use the `dspy.LM` class to set up language models. This replaces the previous client-specific classes. Then, use `dspy.configure` to declare this as the default LM.

For example, to use OpenAI language models, you can do it as follows.

```python
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

### Directly calling the LM

You can simply call the LM with a string to give it a raw prompt, i.e. a string.

```python
lm("hello! this is a raw prompt to GPT-4o-mini")
```

**Output:**
```text
["Hello! It looks like you're trying to interact with a model. How can I assist you today?"]
```

For chat LMs, you can pass a list of messages.

```python
lm(messages=[{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is 2+2?"}])
```

**Output:**
```text
['2 + 2 equals 4.']
```

This is almost never the recommended way to interact with LMs in DSPy, but it is allowed.


### Using the LM with DSPy signatures

You can also use the LM via DSPy [`signature` (input/output spec)](https://dspy-docs.vercel.app/docs/building-blocks/signatures) and [`modules`](https://dspy-docs.vercel.app/docs/building-blocks/modules), which we discuss in more depth in the remaining guides.

```python
# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
```
**Output:**
```text
The castle David Gregory inherited has 7 floors.
```

### Using multiple LMs at once.

The default LM above is GPT-3.5, `gpt3_turbo`. What if I want to run a piece of code with, say, GPT-4 or LLama-2?

Instead of changing the default LM, you can just change it inside a block of code.

:::tip
Using `dspy.configure` and `dspy.context` is thread-safe!
:::

```python
# Run with the default LM configured above, i.e. GPT-3.5
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-3.5:', response.answer)

gpt4_turbo = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=300)

# Run with GPT-4 instead
with dspy.context(lm=gpt4_turbo):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-4-turbo:', response.answer)
```
**Output:**
```text
GPT-3.5: The castle David Gregory inherited has 7 floors.
GPT-4-turbo: The number of floors in the castle David Gregory inherited cannot be determined with the information provided.
```

### Configuring LM attributes

For any LM, you can configure any of the following attributes at initialization or per call.

```python
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

By default LMs in DSPy are cached. If you repeat the same call, you will get the same outputs. But you can turn of caching by setting `cache=False` while declaring `dspy.LM` object.

### Using locally hosted LMs

Any OpenAI-compatible endpoint is easy to set up with an `openai/` prefix as well. This works great for open LMs from HuggingFace hosted locally with SGLang, VLLM, or HF Text-Generation-Inference.

```python
sglang_port = 7501
sglang_url = f"http://localhost:{sglang_port}/v1"
sglang_llama = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct", api_base=sglang_url)

# You could also use text mode, in which the prompts are *not* formatted as messages.
sglang_llama_text = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct", api_base=sglang_url, model_type='text')
```

### Inspecting output and usage metadata

Every LM object maintains the history of its interactions, including inputs, outputs, token usage (and $$$ cost), and metadata.

```python
len(lm.history)  # e.g., 3 calls to the LM

lm.history[-1].keys()  # access the last call to the LM, with all metadata
```

**Output:**
```text
dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost'])
```

## Creating Custom LM Class

## Structured LM output with Adapters

Prompt optimizers in DSPy generate and tune the _instructions_ or the _examples_ in the prompts corresponding to your Signatures. DSPy 2.5 introduces **Adapters** as a layer between Signatures and LMs, responsible for formatting these pieces (Signature I/O fields, instructions, and examples) as well as generating and parsing the outputs. 

In DSPy 2.5, the default Adapters are now more natively aware of chat LMs and are responsible for enforcing types, building on earlier experimental features from `TypedPredictor` and `configure(experimental=True)`. In our experience, this change tends to deliver more consistent pre-optimization quality from various LMs, especially for sophisticated Signatures.

```python
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm, experimental=True)

fact_checking = dspy.ChainOfThought('claims -> verdicts')
fact_checking(claims=["Python was released in 1991.", "Python is a compiled language."])
```

## Defining Custom Adapters