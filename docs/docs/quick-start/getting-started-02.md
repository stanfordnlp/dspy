---
sidebar_position: 3
---

# Getting Started II: Basic RAG

Let's walk through a quick example of **basic retrieval-augmented generation (RAG)** in DSPy. Specifically, let's build **a system for answering Tech questions**, e.g. about Linux or iPhone apps.

Install the latest DSPy via `pip install -U dspy` and follow along. You may also need to install PyTorch via `pip install torch`.

## Continue from Getting Started I.

In [Getting Started I: Basic Question Answering](/quick-start/getting-started-01), we've set up the DSPy LM, loaded some data, and loaded a metric for evaluation.

First, let's download the corpus data that we will use for RAG search. The next cell will seek to download 4 GBs, so it may take a few minutes. A future version of this notebook will come with a cache that allows you to skip downloads and the PyTorch installation.

```python
import os
import requests

urls = [
    'https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json',
    'https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/technology/test_collection.jsonl',
    'https://huggingface.co/dspy/cache/resolve/main/index.pt'
]

for url in urls:
    filename = os.path.basename(url)
    remote_size = int(requests.head(url, allow_redirects=True).headers.get('Content-Length', 0))
    local_size = os.path.getsize(filename) if os.path.exists(filename) else 0

    if local_size != remote_size:
        print(f"Downloading '{filename}'...")
        with requests.get(url, stream=True) as r, open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
```

Having downloaded these items, let's set up the data and other objects from the previous guide.

```python
import ujson
import dspy
from dspy.evaluate import SemanticF1

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

with open('ragqa_arena_tech_500.json') as f:
    data = [dspy.Example(**d).with_inputs('question') for d in ujson.load(f)]
    trainset, valset, devset, testset = data[:50], data[50:150], data[150:300], data[300:500]

metric = SemanticF1()
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=3)
```

## Set up your system's retriever.

As far as DSPy is concerned, you can plug in any Python code for calling tools or retrievers. Hence, for our RAG system, we can plug any tools for the search step. Here, we'll just use OpenAI Embeddings and PyTorch for top-K search, but this is not a special choice, just a convenient one.

```python
import torch
import functools
from litellm import embedding as Embed

with open("test_collection.jsonl") as f:
    corpus = [ujson.loads(line) for line in f]

index = torch.load('index.pt', weights_only=True)
max_characters = 4000 # >98th percentile of document lengths

@functools.lru_cache(maxsize=None)
def search(query, k=5):
    query_embedding = torch.tensor(Embed(input=query, model="text-embedding-3-small").data[0]['embedding'])
    topk_scores, topk_indices = torch.matmul(index, query_embedding).topk(k)
    topK = [dict(score=score.item(), **corpus[idx]) for idx, score in zip(topk_indices, topk_scores)]
    return [doc['text'][:max_characters] for doc in topK]
```

## Build your first RAG `Module`.

In the previous guide, we looked at individual DSPy modules in isolation, e.g. `dspy.Predict("question -> answer")`.

What if we want to build a DSPy _program_ that has multiple steps? The syntax below with `dspy.Module` allows you to connect a few pieces together, in this case, our retriever and a generation module, so the whole system can be optimized.

Concretely, in the `__init__` method, you declare any sub-module you'll need, which in this case is just a `dspy.ChainOfThought('context, question -> response')` module that takes retrieved context, a question, and produces a response. In the `forward` method, you simply express any Python control flow you like, possibly using your modules. In this case, we first invoke the `search` function defined earlier and then invoke the `self.respond` ChainOfThought module.

```python
class RAG(dspy.Module):
    def __init__(self, num_docs=5):
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question, k=self.num_docs)
        return self.respond(context=context, question=question)
```

Let's use the RAG module.

```
rag = RAG()
rag(question="what are high memory and low memory on linux?")
```

**Output:**
```
Prediction(
    reasoning="High memory and low memory in Linux refer to the organization of memory in the system, particularly in the context of the Linux kernel's virtual memory management. High memory is the portion of physical memory that is not directly mapped by the kernel's page tables, meaning that user-space applications cannot access it directly. Low memory, on the other hand, is the part of memory that the kernel can access directly. In a typical 32-bit architecture, the virtual memory is split into 3 GB for user space (low memory) and 1 GB for kernel space (high memory). The distinction is important for memory management, especially when dealing with physical memory that cannot be mapped contiguously. Understanding this split is crucial for developers working with the Linux kernel, as it affects how memory is allocated and accessed.",
    response="In Linux, high memory refers to the portion of physical memory that is not directly mapped by the kernel's page tables, making it inaccessible to user-space applications. Low memory is the segment that the kernel can access directly. In a typical 32-bit architecture, the memory is divided into 3 GB for user space (low memory) and 1 GB for kernel space (high memory). This organization is essential for efficient memory management and affects how the kernel interacts with physical memory, especially in scenarios where contiguous memory is required."
)
```

```python
dspy.inspect_history()
```

**Output:**     
See this [gist](https://gist.github.com/okhat/d807032e138862bb54616dcd2f4d481c).


In the previous guide with a CoT module, we got nearly 40% in terms of semantic F1 on our `devset`. Would this `RAG` module score better?


```python
evaluate(RAG())
```

**Output:**
```
Average Metric: 79.44815807647615 / 150  (53.0): 100%|██████████| 150/150 [00:05<00:00, 29.11it/s]
```

| | question | example_response | reasoning | pred_response | SemanticF1 |
|---|---|---|---|---|---|
| 0 | why is mercurial considered to be easier than git? | Mercurial's syntax is considered more familiar, especially for those accustomed to SVN, and is well documented. It focuses on interface aspects, which initially makes learning... | Mercurial is considered easier than Git for several reasons. Firstly, its syntax is more familiar to users coming from Subversion (SVN), making the transition smoother.... | Mercurial is considered easier than Git primarily because of its more familiar syntax for users transitioning from systems like Subversion (SVN), its straightforward workflow without... | ✔️ [0.797] |
| 1 | open finder window from current terminal location? | If you type 'open .' in Terminal, it will open the current directory in a Finder window. Alternatively, you can execute the command open `pwd`... | To open a Finder window from the current terminal location, you can use the command `open .` in the terminal. This command will open a... | You can open a Finder window from your current terminal location by typing the command `open .` in the terminal. This will launch a Finder... | ✔️ [0.857] |
| 2 | how to import secret gpg key (copied from one machine to another)? | It is advised that it is necessary to add `--import` to the command line to import the private key and that according to the man... | To import a secret GPG key that has been copied from one machine to another, you typically need to use the `gpg --import` command. The... | To import a secret GPG key that you have copied from one machine to another, follow these steps: 1. On the original machine, export your... | |


## Using a DSPy `Optimizer` to improve your RAG prompt.

Off the shelf, our `RAG` module scores 53%. What are our options to make it stronger? One of the various choices DSPy offers is optimizing the prompts in our pipeline.

If there are many sub-modules in your program, all of them will be optimized together. In this case, there's only one: `self.respond = dspy.ChainOfThought('context, question -> response')`

Let's set up and use DSPy's [MIPRO (v2) optimizer](/deep-dive/optimizers/miprov2). The run below has a cost around $1.5 (for the `medium` auto setting) and may take some 20-30 minutes depending on your number of threads.


```python
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)  # use fewer threads if your rate limit is small

optimized_rag = tp.compile(RAG(), trainset=trainset, valset=valset,
                           max_bootstrapped_demos=2, max_labeled_demos=2,
                           requires_permission_to_run=False)
```

**Output:**     
See this [gist](https://gist.github.com/okhat/d6606e480a94c88180441617342699eb).


The prompt optimization process here is pretty systematic, you can learn about it for example in this paper. Importantly, it's not a magic button. It's very possible that it can overfit your training set for instance and not generalize well to a held-out set, making it essential that we iteratively validate our programs.

Let's check on an example here, asking the same question to the baseline `rag = RAG()` program, which was not optimized, and to the `optimized_rag = MIPROv2(..)(..)` program, after prompt optimization.


```python
baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
print(baseline.response)
```

**Output:**
```
You are correct; cmd+Tab does not activate hidden or minimized windows in macOS. It functions as an application switcher, allowing you to switch between open applications, but it does not bring up minimized windows. To access minimized windows, you would need to click on them directly or use other shortcuts.
```


```python
pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
print(pred.response)
```

**Output:**
```
In macOS, the Command+Tab shortcut is specifically designed to switch between applications rather than individual windows. This means that if an application is minimized or hidden, it will not appear in the Command+Tab application switcher. Therefore, you cannot use Command+Tab to access minimized or hidden windows directly.

If you want to bring a minimized window back into view, you can click on the application's icon in the Dock, or you can use the Command+M shortcut to minimize the current window. For switching between windows of the same application, you can use Command+` (the backtick key) to cycle through open windows of the active application.

For users who prefer a behavior similar to Windows, where minimized windows can be accessed through a single shortcut, third-party applications like HyperSwitch or Witch can provide additional functionality to manage window switching more effectively.
```

You can use `dspy.inspect_history(n=2)` to view the RAG prompt [before optimization](https://gist.github.com/okhat/5d04648f2226e72e66e26a8cb1456ee4) and [after optimization](https://gist.github.com/okhat/79405b8889b4b07da577ee19f1a3479a).

Concretely, the optimized prompt:

1. Constructs the following instruction,
```
Using the provided `context` and `question`, analyze the information step by step to generate a comprehensive and informative `response`. Ensure that the response clearly explains the concepts involved, highlights key distinctions, and addresses any complexities noted in the context.
```

2. And includes two fully worked out RAG examples with synthetic reasoning and answers, e.g. `how to transfer whatsapp voice message to computer?`.

Let's now evaluate on the overall devset.


```python
evaluate(optimized_rag)
```

**Output:**
```
Average Metric: 92.16999654981839 / 150  (61.4): 100%|██████████| 150/150 [00:00<00:00, 399.21it/s]
```


| | question | example_response | reasoning | pred_response | SemanticF1 |
|---|---|---|---|---|---|
| 0 | why is mercurial considered to be easier than git? | Mercurial's syntax is considered more familiar, especially for those accustomed to SVN, and is well documented. It focuses on interface aspects, which initially makes learning... | Mercurial is often considered easier than Git due to its user-friendly design and interface, which is particularly appealing to those new to version control systems... | Mercurial is considered easier than Git for several reasons: 1. **Familiar Syntax**: Mercurial's command syntax is often seen as more intuitive, especially for users coming... | ✔️ [0.874] |
| 1 | open finder window from current terminal location? | If you type 'open .' in Terminal, it will open the current directory in a Finder window. Alternatively, you can execute the command open `pwd`... | To open a Finder window from the current terminal location on a Mac, there are several methods available. The simplest way is to use the... | To open a Finder window from your current terminal location on a Mac, you can use the following methods: 1. **Using Terminal Command**: - Simply... | ✔️ [0.333] |
| 2 | how to import secret gpg key (copied from one machine to another)? | It is advised that it is necessary to add `--import` to the command line to import the private key and that according to the man... | To import a secret GPG key that has been copied from one machine to another, it is essential to follow a series of steps that... | To import a secret GPG key that you have copied from one machine to another, follow these steps: 1. **Export the Secret Key from the... | |


## Keeping an eye on cost.

DSPy allows you to track the cost of your programs, which can be used to monitor the cost of your calls. Here, we'll show you how to track the cost of your programs with DSPy.

```python
sum([x['cost'] for x in lm.history if x['cost'] is not None])  # in USD, as calculated by LiteLLM for certain providers
```

## Saving and loading.

The optimized program has a pretty simple structure on the inside. Feel free to explore it.

Here, we'll save `optimized_rag` so we can load it again later without having to optimize from scratch.

```python
optimized_rag.save("optimized_rag.json")

loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")

loaded_rag(question="cmd+tab does not work on hidden or minimized windows")
```

**Output:**
```
Prediction(
    reasoning='The behavior of the Command+Tab shortcut in macOS is designed to switch between applications rather than individual windows. When an application is minimized or hidden, it does not appear in the application switcher, which is why Command+Tab does not work for those windows. Understanding this limitation is important for users who expect similar functionality to that found in other operating systems, such as Windows, where Alt+Tab can switch between all open windows, including minimized ones.',
    response="In macOS, the Command+Tab shortcut is specifically designed to switch between applications rather than individual windows. This means that if an application is minimized or hidden, it will not appear in the Command+Tab application switcher. Therefore, you cannot use Command+Tab to access minimized or hidden windows directly.\n\nIf you want to bring a minimized window back into view, you can click on the application's icon in the Dock, or you can use the Command+M shortcut to minimize the current window. For switching between windows of the same application, you can use Command+` (the backtick key) to cycle through open windows of the active application.\n\nFor users who prefer a behavior similar to Windows, where minimized windows can be accessed through a single shortcut, third-party applications like HyperSwitch or Witch can provide additional functionality to manage window switching more effectively."
)
```

## What's next?

Improving from just below 40% to above 60% on this task, in terms of `SemanticF1`, was pretty easy.

But DSPy gives you paths to continue iterating on the quality of your system and we have barely scratched the surface.

In general, you have the following tools:

1. Explore better system architectures for your program, e.g. what if we ask the LM to generate search queries for the retriever? See this [notebook](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb) or the [STORM pipeline](https://arxiv.org/abs/2402.14207) built in DSPy.
2. Explore different [prompt optimizers](https://arxiv.org/abs/2406.11695) or [weight optimizers](https://arxiv.org/abs/2407.10930). See the **[Optimizers Docs](/building-blocks/6-optimizers)**.
3. Scale inference time compute using DSPy Optimizers, e.g. this [notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/agents/multi_agent.ipynb).
4. Cut cost by distilling to a smaller LM, via prompt or weight optimization, e.g. [this notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/nli/scone/scone.ipynb) or [this notebook](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/qa/hotpot/multihop_finetune.ipynb).

How do you decide which ones to proceed with first?

The first step is to look at your system outputs, which will allow you to identify the sources of lower performance if any. While doing all of this, make sure you continue to refine your metric, e.g. by optimizing against your judgments, and to collect more (or more realistic) data, e.g. from related domains or from putting a demo of your system in front of users.

Learn more about the [development cycle](/building-blocks/solving_your_task) in DSPy.
