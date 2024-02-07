---
sidebar_position: 4
---

# Language Models

This guide assumes you followed the [intro tutorial](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb) to build your first few DSPy programs.

Remember that a **DSPy program** is just Python code that calls one or more DSPy modules, like `dspy.Predict` or `dspy.ChainOfThought`, to use LMs.

## 1) Short Intro to LMs in DSPy {#1-short-intro-to-lms-in-dspy}

``` python
# Install `dspy-ai` if needed.

try: import dspy
except ImportError:
    %pip install dspy-ai
    import dspy
```

## 2) Supported LM clients. {#2-supported-lm-clients}

### Remote LMs. {#remote-lms}

These models are managed services. You just need to sign up and obtain
an API key.

1.  `dspy.OpenAI` for GPT-3.5 and GPT-4.

2.  `dspy.Cohere`

3.  `dspy.Anyscale` for hosted Llama2 models.

### Local LMs. {#local-lms}

You need to host these models on your own GPU(s). Below, we include
pointers for how to do that.

1.  `dspy.HFClientTGI`: for HuggingFace models through the Text Generation Inference (TGI) system. [Tutorial: How do I install and launch the TGI server?](/api/hosting_language_models_locally/TGI)

2.  `dspy.HFClientVLLM`: for HuggingFace models through vLLM. [Tutorial: How do I install and launch the vLLM server?](/api/hosting_language_models_locally/vLLM)

3.  `dspy.HFModel` (experimental)

4.  `dspy.Ollama` (experimental)

5.  `dspy.ChatModuleClient` (experimental): [How do I install and use MLC?](/api/hosting_language_models_locally/MLC)

If there are other clients you want added, let us know!

## 3) Setting up the LM client. {#3-setting-up-the-lm-client}

You can just call the constructor that connects to the LM. Then, use
`dspy.configure` to declare this as the default LM.

For example, for OpenAI, you can do it as follows.

``` python
# TODO: Add a graceful line for OPENAI_API_KEY.

gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=300)
gpt4_turbo = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=300)

# cohere = dspy.Cohere(...)
# anyscale = dspy.Anyscale(...)
# tgi_llama2 = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")

dspy.configure(lm=gpt3_turbo)
```

## 4) Using a different LM within a code block. {#4-using-a-different-lm-within-a-code-block}

The default LM above is GPT-3.5, `gpt3_turbo`. What if I want to run a
piece of code with, say, GPT-4 or LLama-2?

Instead of changing the default LM, you can just change it inside a
block of code.

**Tip:** Using `dspy.configure` and `dspy.context` is thread-safe!

``` python
qa = dspy.ChainOfThought('question -> answer')

response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)

with dspy.context(lm=gpt4_turbo):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print(response.answer)
```

    The castle David Gregory inherited has 7 floors.
    The number of floors in the castle David Gregory inherited cannot be determined with the information provided.

## 5) Tips and Tricks. {#5-tips-and-tricks}

In DSPy, all LM calls are cached. If you repeat the same call, you will
get the same outputs. (If you change the inputs or configurations, you
will get new outputs.)

To generate 5 outputs, you can use `n=5` in the module constructor, or
pass `config=dict(n=5)` when invoking the module.

``` python
qa = dspy.ChainOfThought('question -> answer', n=5)

response = qa(question="How many floors are in the castle David Gregory inherited?")
response.completions.answer
```

    ["The specific number of floors in David Gregory's inherited castle is not provided here, so further research would be needed to determine the answer.",
     'The castle David Gregory inherited has 4 floors.',
     'The castle David Gregory inherited has 5 floors.',
     'David Gregory inherited 10 floors in the castle.',
     'The castle David Gregory inherited has 5 floors.']

If you just call `qa(...)` in a loop with the same input, it will always
return the same value! That\'s by design.

To loop and generate one output at a time with the same input, bypass
the cache by making sure each request is (slightly) unique, as below.

``` python
for idx in range(5):
    response = qa(question="How many floors are in the castle David Gregory inherited?", config=dict(temperature=0.7+0.0001*idx))
    print(response.answer)
```

    The specific number of floors in David Gregory's inherited castle is not provided here, so further research would be needed to determine the answer.
    It is not possible to determine the exact number of floors in the castle David Gregory inherited without specific information about the castle's layout and history.
    The castle David Gregory inherited has 5 floors.
    We need more information to determine the number of floors in the castle David Gregory inherited.
    The castle David Gregory inherited has a total of 6 floors.
