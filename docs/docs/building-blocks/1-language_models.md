---
sidebar_position: 2
---

# Language Models

The most powerful features in DSPy revolve around algorithmically optimizing the prompts (or weights) of LMs, especially when you're building programs that use the LMs within a pipeline.

Let's first make sure you can set up your language model. DSPy support clients for many remote and local LMs.

## Setting up the LM client.

You can just call the constructor that connects to the LM. Then, use `dspy.configure` to declare this as the dexfault LM.

For example, to use OpenAI language models, you can do it as follows.

```python
gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=300)
dspy.configure(lm=gpt3_turbo)
```

## Directly calling the LM.

You can simply call the LM with a string to give it a raw prompt, i.e. a string.

```python
gpt3_turbo("hello! this is a raw prompt to GPT-3.5")
```

**Output:**
```text
['Hello! How can I assist you today?']
```

This is almost never the recommended way to interact with LMs in DSPy, but it is allowed.

## Using the LM with DSPy signatures.

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

## Using multiple LMs at once.

The default LM above is GPT-3.5, `gpt3_turbo`. What if I want to run a piece of code with, say, GPT-4 or LLama-2?

Instead of changing the default LM, you can just change it inside a block of code.

**Tip:** Using `dspy.configure` and `dspy.context` is thread-safe!

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

## Tips and Tricks.

In DSPy, all LM calls are cached. If you repeat the same call, you will
get the same outputs. (If you change the inputs or configurations, you
will get new outputs.)

To generate 5 outputs, you can use `n=5` in the module constructor, or
pass `config=dict(n=5)` when invoking the module.

```python
qa = dspy.ChainOfThought('question -> answer', n=5)

response = qa(question="How many floors are in the castle David Gregory inherited?")
response.completions.answer
```
**Output:**
```text
["The specific number of floors in David Gregory's inherited castle is not provided here, so further research would be needed to determine the answer.",
    'The castle David Gregory inherited has 4 floors.',
    'The castle David Gregory inherited has 5 floors.',
    'David Gregory inherited 10 floors in the castle.',
    'The castle David Gregory inherited has 5 floors.']
```

If you just call `qa(...)` in a loop with the same input, it will always
return the same value! That\'s by design.

To loop and generate one output at a time with the same input, bypass
the cache by making sure each request is (slightly) unique, as below.

```python
for idx in range(5):
    response = qa(question="How many floors are in the castle David Gregory inherited?", config=dict(temperature=0.7+0.0001*idx))
    print(f'{idx+1}.', response.answer)
```
**Output:**
```text
1. The specific number of floors in David Gregory's inherited castle is not provided here, so further research would be needed to determine the answer.
2. It is not possible to determine the exact number of floors in the castle David Gregory inherited without specific information about the castle's layout and history.
3. The castle David Gregory inherited has 5 floors.
4. We need more information to determine the number of floors in the castle David Gregory inherited.
5. The castle David Gregory inherited has a total of 6 floors.
```

## Remote LMs.

These models are managed services. You just need to sign up and obtain an API key. Calling any of the remote LMs below assumes authentication and mirrors the following format for setting up the LM:

```python
lm = dspy.{provider_listed_below}(model="your model", model_request_kwargs="...")
```

1.  `dspy.OpenAI` for GPT-3.5 and GPT-4.

2.  `dspy.Cohere`

3.  `dspy.Anyscale` for hosted Llama2 models.

4. `dspy.Together` for hosted various open source models.


### Local LMs.

You need to host these models on your own GPU(s). Below, we include pointers for how to do that.

1.  `dspy.HFClientTGI`: for HuggingFace models through the Text Generation Inference (TGI) system. [Tutorial: How do I install and launch the TGI server?](https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/local_models/HFClientTGI)

```python
tgi_llama2 = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
```

2.  `dspy.HFClientVLLM`: for HuggingFace models through vLLM. [Tutorial: How do I install and launch the vLLM server?](https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/local_models/HFClientVLLM)

```python
vllm_llama2 = dspy.HFClientVLLM(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
```

3.  `dspy.HFModel` (experimental) [Tutorial: How do I initialize models using HFModel](https://dspy-docs.vercel.app/api/local_language_model_clients/HFModel)

```python
llama = dspy.HFModel(model = 'meta-llama/Llama-2-7b-hf')
```

4.  `dspy.Ollama` (experimental) for open source models through [Ollama](https://ollama.com). [Tutorial: How do I install and use Ollama on a local computer?](https://dspy-docs.vercel.app/api/local_language_model_clients/Ollama)\n",

```python
mistral_ollama = dspy.OllamaLocal(model='mistral')
```

5.  `dspy.ChatModuleClient` (experimental): [How do I install and use MLC?](https://dspy-docs.vercel.app/api/local_language_model_clients/MLC)

```python
model = 'dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1'
model_path = 'dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so'

llama = dspy.ChatModuleClient(model=model, model_path=model_path)
```
