---
sidebar_position: 3
---

# Modules

A **DSPy module** is a building block for programs that use LMs.

- Each built-in module abstracts a **prompting technique** (like chain of thought or ReAct). Crucially, they are generalized to handle any [DSPy Signature]().

- A DSPy module has **learnable parameters** (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.


## How do I use a built-in module?

We'll assume you are already at least a little familiar with [DSPy signatures], which are declarative specs for defining the behavior of any module we use in DSPy.

