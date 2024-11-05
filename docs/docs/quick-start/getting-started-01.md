---
sidebar_position: 2
---

# Getting Started I: Basic Question Answering

Let's walk through a quick example of **basic question answering** in DSPy. Specifically, let's build **a system for answering Tech questions**, e.g. about Linux or iPhone apps.

Install the latest DSPy via `pip install -U dspy` and follow along. If you're looking instead for a conceptual overview of DSPy, this [recent lecture](https://www.youtube.com/live/JEMYuzrKLUw) is a good place to start.

## Configuring the DSPy environment

Let's tell DSPy that we will use OpenAI's `gpt-4o-mini` in our modules. To authenticate, DSPy will look into your `OPENAI_API_KEY`. You can easily swap this out for [other providers or local models](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb).

```python
import dspy

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

## Exploring some basic DSPy `Module`s.

You can always prompt the LM directly via `lm(prompt="prompt")` or `lm(messages=[...])`. However, DSPy gives you `Modules` as a better way to define your LM functions.

The simplest module is `dspy.Predict`. It takes a [DSPy Signature](/building-blocks/2-signatures), i.e. a structured input/output schema, and gives you back a callable function for the behavior you specified. Let's use the "in-line" notation for signatures to declare a module that takes a `question` (of type `str`) as input and produces a `response` as an output.

```python
qa = dspy.Predict('question: str -> response: str')
qa(question="what are high memory and low memory on linux?").response
```

**Output:**
```
'In Linux, "high memory" and "low memory" refer to different regions of the system\'s memory address space, particularly in the context of 32-bit architectures.\n\n- **Low Memory**: This typically refers to the first 896 MB of memory in a 32-bit system. It is directly accessible by the kernel and is used for kernel data structures and user processes. The low memory region is where most of the system\'s memory management occurs, and it is where the kernel can allocate memory for processes without needing special handling.\n\n- **High Memory**: This refers to memory above the 896 MB threshold in a 32-bit system. The kernel cannot directly access this memory without special mechanisms because of the limitations of the 32-bit address space. High memory is used for user processes that require more memory than what is available in the low memory region. The kernel can manage high memory through techniques like "highmem" support, which allows it to map high memory pages into the kernel\'s address space when needed.\n\nIn summary, low memory is directly accessible by the kernel, while high memory requires additional handling for the kernel to access it, especially in 32-bit systems. In 64-bit systems, this distinction is less relevant as the addressable memory space is significantly larger.'
```

Notice how the variable names we specified in the signature defined our input and output argument names and their role.

Now, what did DSPy do to build this `qa` module? Nothing fancy in this example, yet. The module passed your signature, LM, and inputs to an [Adapter](/building-blocks/1-language_models#structured-lm-output-with-adapters), which is a layer that handles structuring the inputs and parsing structured outputs to fit your signature.

Let's see it directly. You can inspect the `n` last prompts sent by DSPy easily.

```python
dspy.inspect_history(n=1)
```

**Output:**   
See this [gist](https://gist.github.com/okhat/aff3c9788ccddf726fdfeb78e40e5d22).


DSPy has various built-in modules, e.g. `dspy.ChainOfThought`, `dspy.ProgramOfThought`, and `dspy.ReAct`. These are interchangeable with basic `dspy.Predict`: they take your signature, which is specific to your task, and they apply general-purpose prompting techniques and inference-time strategies to it.

For example, `dspy.ChainOfThought` is an easy way to elicit `reasoning` out of your LM before it commits to the outputs requested in your signature.

In the example below, we'll omit `str` types (as the default type is string). You should feel free to experiment with other fields and types, e.g. try `topics: list[str]` or `is_realistic: bool`.

```python
cot = dspy.ChainOfThought('question -> response')
cot(question="should curly braces appear on their own line?")
```

**Output:**
```
Prediction(
    reasoning="The placement of curly braces on their own line is largely a matter of coding style and conventions. In some programming languages and style guides, such as those used in C, C++, and Java, it is common to place opening curly braces on the same line as the control statement (like `if`, `for`, etc.) and closing braces on a new line. However, other styles, such as the Allman style, advocate for placing both opening and closing braces on their own lines. Ultimately, the decision should be based on the team's coding standards or personal preference, as long as it maintains readability and consistency throughout the code.",
    response="Curly braces can either appear on their own line or not, depending on the coding style you choose to follow. It's important to be consistent with whichever style you adopt."
)
```


Interestingly, asking for reasoning made the output `response` shorter in this case. Is this a good thing or a bad thing? It depends on what you need: there's no free lunch, but DSPy gives you the tools to experiment with different strategies extremely quickly.

By the way, `dspy.ChainOfThought` is implemented in DSPy, using `dspy.Predict`. This is a good place to `dspy.inspect_history` if you're curious.

## Using DSPy well involves evaluation and iterative development.

You already know a lot about DSPy at this point. If all you want is quick scripting, this much of DSPy already enables a lot. Sprinkling DSPy signatures and modules into your Python control flow is a pretty ergonomic way to just get stuff done with LMs.

That said, you're likely here because you want to build a high-quality system and improve it over time. The way to do that in DSPy is to iterate fast by evaluating the quality of your system and using DSPy's powerful tools, e.g. [Optimizers](/building-blocks/6-optimizers). You can learn about the [appropriate development cycle in DSPy here](/building-blocks/solving_your_task).

## Manipulating `Example`s in DSPy.

To measure the quality of your DSPy system, you need (1) a bunch of input values, like `question`s for example, and (2) a `metric` that can score the quality of an output from your system. Metrics vary widely. Some metrics need ground-truth labels of ideal outputs, e.g. for classification or question answering. Other metrics are self-supervised, e.g. checking faithfulness or lack of hallucination, perhaps using a DSPy program as a judge of these qualities.

Let's load a dataset of questions and their (pretty long) gold answers. Since we started this notebook with the goal of building **a system for answering Tech questions**, we obtained a bunch of StackExchange-based questions and their correct answers from the [RAG-QA Arena](https://arxiv.org/abs/2407.13998) dataset. (Learn more about the [development cycle](/building-blocks/solving_your_task) if you don't have data for your task.)


```python
import ujson

# Download 500 question--answer pairs from the RAG-QA Arena "Tech" dataset.
!wget https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json

with open('ragqa_arena_tech_500.json') as f:
    data = ujson.load(f)

# Inspect one datapoint.
data[0]
```

**Output:**
```
{'question': 'how to transfer whatsapp voice message to computer?',
  'response': 'To transfer voice notes from WhatsApp on your device to your computer, you have the option to select the "Share" feature within the app and send the files via Email, Gmail, Bluetooth, or other available services.  \nYou can also move the files onto your phone\'s SD card, connect your phone to your computer via a USB cable, then find and transfer the files via File Explorer on your PC. \nAlternatively, you can choose to attach all the desired voice notes to an email and, from your phone, send them to your own email address.  \nUpon receiving the email on your computer, you can then download the voice note attachments.'}
```

Given a simple dict like this, let's create a list of `dspy.Example`s, which is the datatype that carries training (or test) datapoints in DSPy.

When you build a `dspy.Example`, you should generally specify `.with_inputs("field1", "field2", ...)` to indicate which fields are inputs. The other fields are treated as labels or metadata.

```python
data = [dspy.Example(**d).with_inputs('question') for d in data]

# Let's pick an `example` here from the data.
example = data[2]
example
```

**Output:**
```
Example({'question': 'what are high memory and low memory on linux?', 'response': '"High Memory" refers to the application or user space, the memory that user programs can use and which isn\'t permanently mapped in the kernel\'s space, while "Low Memory" is the kernel\'s space, which the kernel can address directly and is permanently mapped. \nThe user cannot access the Low Memory as it is set aside for the required kernel programs.'}) (input_keys={'question'})
```

Now, let's divide the data into:

- Training and Validation sets:
    - These are the splits you typically give to DSPy optimizers.
    - Optimizers typically learn directly from the training examples and check their progress using the validation examples.
    - It's good to have 30--300 examples for training and validation each.
    - For prompt optimizers in particular, it's often better to pass _more_ validation than training.

- Development and Test sets: The rest, typically on the order of 30--1000, can be used for:
    - development (i.e., you can inspect them as you iterate on your system) and
    - testing (final held-out evaluation).


```python
trainset, valset, devset, testset = data[:50], data[50:150], data[150:300], data[300:500]

len(trainset), len(valset), len(devset), len(testset)
```

**Output:**
```
(50, 100, 150, 200)
```


## Evaluation in DSPy.

What kind of metric can suit our question-answering task? There are many choices, but since the answers are long, we may ask: How well does the system response _cover_ all key facts in the gold response? And the other way around, how well is the system response _not saying things_ that aren't in the gold response?

That metric is essentially a **semantic F1**, so let's load a `SemanticF1` metric from DSPy. This metric is actually implemented as a [very simple DSPy module](https://github.com/stanfordnlp/dspy/blob/77c2e1cceba427c7f91edb2ed5653276fb0c6de7/dspy/evaluate/auto_evaluation.py#L21) using whatever LM we're working with.


```python
from dspy.evaluate import SemanticF1

# Instantiate the metric.
metric = SemanticF1()

# Produce a prediction from our `cot` module, using the `example` above as input.
pred = cot(**example.inputs())

# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score:.2f}")
```

**Output:**
```
Question: 	 what are high memory and low memory on linux?

Gold Response: 	 "High Memory" refers to the application or user space, the memory that user programs can use and which isn't permanently mapped in the kernel's space, while "Low Memory" is the kernel's space, which the kernel can address directly and is permanently mapped. 
The user cannot access the Low Memory as it is set aside for the required kernel programs.

Predicted Response: 	 In Linux, "low memory" refers to the memory that is directly accessible by the kernel and user processes, typically the first 4GB on a 32-bit system. "High memory" refers to memory above this limit, which is not directly accessible by the kernel in a 32-bit environment. This distinction is crucial for memory management, particularly in systems with large amounts of RAM, as it influences how memory is allocated and accessed.

Semantic F1 Score: 0.80
```

The final DSPy module call above actually happens inside `metric`. You might be curious how it measured the semantic F1 for this example.


```python
dspy.inspect_history(n=1)
```

**Output:**     
See this [gist](https://gist.github.com/okhat/57bf86472d1e14812c0ae33fba5353f8).

For evaluation, you could use the metric above in a simple loop and just average the score. But for nice parallelism and utilities, we can rely on `dspy.Evaluate`.

```python
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=3)

# Evaluate the Chain-of-Thought program.
evaluate(cot)
```

**Output:**
```
Average Metric: 59.565342393613165 / 150  (39.7): 100%|██████████| 150/150 [00:00<00:00, 432.92it/s]
```

The table you'll get in the output would look like:

| | question | example_response | reasoning | pred_response | SemanticF1 |
|---|---|---|---|---|---|
| 0 | why is mercurial considered to be easier than git? | Mercurial's syntax is considered more familiar, especially for those accustomed to SVN, and is well documented. It focuses on interface aspects, which initially makes learning... | Mercurial is often considered easier than Git for several reasons. Firstly, Mercurial has a simpler command structure and a more consistent user interface, which can... | Mercurial is considered easier than Git primarily due to its simpler command structure and more consistent user interface, making it more approachable for beginners. Its... | ✔️ [0.545] |
| 1 | open finder window from current terminal location? | If you type 'open .' in Terminal, it will open the current directory in a Finder window. Alternatively, you can execute the command open `pwd`... | To open a Finder window from the current terminal location on a Mac, you can use the `open` command followed by a dot (`.`) which... | You can open a Finder window from your current terminal location by using the following command:\n```\nopen .\n``` | ✔️ [0.667] |
| 2 | how to import secret gpg key (copied from one machine to another)? | It is advised that it is necessary to add `--import` to the command line to import the private key and that according to the man... | To import a secret GPG key that has been copied from one machine to another, you need to ensure that the key is in the... | To import a secret GPG key that you have copied from one machine to another, follow these steps: 1. **Transfer the Key**: Ensure that the... | ✔️ [0.708] |

## What's next?

In this guide, we built a very simple chain-of-thought module for question answering and evaluated it on a small dataset.

Can we do better? In the next guide, we will build a retrieval-augmented generation (RAG) program in DSPy for the same task.

We'll see how this can boost the score substantially, then we'll use one of the DSPy Optimizers to _compile_ our RAG program to higher-quality prompts, raising our scores even more.

Continue here. [Getting Started II: An Example for Basic RAG](/quick-start/getting-started-02)
