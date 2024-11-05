---
sidebar_position: 3
---

# Modules

A **DSPy module** is a building block for programs that use LMs.

- Each built-in module abstracts a **prompting technique** (like chain of thought or ReAct). Crucially, they are generalized to handle any [DSPy Signature](/building-blocks/2-signatures).

- A DSPy module has **learnable parameters** (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.


## How do I use a built-in module, like `dspy.Predict` or `dspy.ChainOfThought`?

Let's start with the most fundamental module, `dspy.Predict`. Internally, all other DSPy modules are just built using `dspy.Predict`.

We'll assume you are already at least a little familiar with [DSPy signatures](/building-blocks/2-signatures), which are declarative specs for defining the behavior of any module we use in DSPy.

To use a module, we first **declare** it by giving it a signature. Then we **call** the module with the input arguments, and extract the output fields!

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment')

# 2) Call with input argument(s). 
response = classify(sentence=sentence)

# 3) Access the output.
print(response.sentiment)
```
**Output:**
```text
Positive
```

When we declare a module, we can pass configuration keys to it.

Below, we'll pass `n=5` to request five completions. We can also pass `temperature` or `max_len`, etc.

Let's use `dspy.ChainOfThought`. In many cases, simply swapping `dspy.ChainOfThought` in place of `dspy.Predict` improves quality.

```python
question = "What's something great about the ColBERT retrieval model?"

# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)

# 2) Call with input argument.
response = classify(question=question)

# 3) Access the outputs.
response.completions.answer
```
**Output:**
```text
['One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.',
 'Its ability to efficiently retrieve relevant information from large document collections.',
 'One great thing about the ColBERT retrieval model is its superior performance compared to other models and its efficient use of pre-trained language models.',
 'One great thing about the ColBERT retrieval model is its superior efficiency and accuracy compared to other models.',
 'One great thing about the ColBERT retrieval model is its ability to incorporate user feedback and support complex queries.']
```

Let's discuss the output object here.

The `dspy.ChainOfThought` module will generally inject a `reasoning` before the output field(s) of your signature.

Let's inspect the (first) reasoning and answer!

```python
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```
**Output:**
```text
Rationale: produce the answer. We can consider the fact that ColBERT has shown to outperform other state-of-the-art retrieval models in terms of efficiency and effectiveness. It uses contextualized embeddings and performs document retrieval in a way that is both accurate and scalable.
Answer: One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.
```

This is accessible whether we request one or many completions.

We can also access the different completions as a list of `Prediction`s or as several lists, one for each field.

```python
response.completions[3].reasoning == response.completions.reasoning[3]
```
**Output:**
```text
True
```


## What other DSPy modules are there? How can I use them?

The others are very similar. They mainly change the internal behavior with which your signature is implemented!

1. **`dspy.Predict`**: Basic predictor. Does not modify the signature. Handles the key forms of learning (i.e., storing the instructions and demonstrations and updates to the LM).

2. **`dspy.ChainOfThought`**: Teaches the LM to think step-by-step before committing to the signature's response.

3. **`dspy.ProgramOfThought`**: Teaches the LM to output code, whose execution results will dictate the response.

4. **`dspy.ReAct`**: An agent that can use tools to implement the given signature.

5. **`dspy.MultiChainComparison`**: Can compare multiple outputs from `ChainOfThought` to produce a final prediction.


We also have some function-style modules:

6. **`dspy.majority`**: Can do basic voting to return the most popular response from a set of predictions.


Check out further examples in [each module's respective guide](/deep-dive/modules/guide).


## How do I compose multiple modules into a bigger program?

DSPy is just Python code that uses modules in any control flow you like. (There's some magic internally at `compile` time to trace your LM calls.)

What this means is that, you can just call the modules freely. No weird abstractions for chaining calls.

This is basically PyTorch's design approach for define-by-run / dynamic computation graphs. Refer to the intro tutorials for examples.
