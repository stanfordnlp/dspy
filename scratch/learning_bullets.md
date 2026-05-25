# Learning Bullets for "Getting Started with DSPy"

## 1. Installing DSPy and connecting to a language model

- How to install DSPy using `pip` or `uv` in a Python 3.9+ environment
- How to configure a language model with `dspy.LM` and `dspy.configure`
- How to manually call the configured LM to verify the connection works

## 2. Writing our first DSPy program

- What a **Signature** is and how it defines the input/output contract for a task
- What a **Module** is and how `dspy.Predict` turns a signature into a runnable program
- How DSPy handles prompt generation, templating, and output parsing automatically

## 3. Expanding your signature: more inputs and adding types

- How to add multiple inputs and outputs by separating field names with commas
- How mindful field naming guides the model's behavior and improves output quality
- How typing fields (e.g., `bool`, `list[str]`) enables output coercion and catches type mismatches

## 4. Writing a class-based signature

- How to write class-based signatures using `dspy.Signature` with docstrings and field descriptions
- How to pass class-based signatures to modules just like string signatures
- How richer types like `Literal` constrain outputs and validate inputs at call time

## 5. Change inference strategies by changing the module

- How `dspy.ChainOfThought` prompts the model to reason before producing a final answer
- How to access the model's reasoning via the `result.reasoning` attribute
- That different modules define different test-time strategies while reusing the same signature

## 6. Give the program tools with `dspy.ReAct`

- How tools are standard Python functions with type hints and docstrings
- How to define an agent with `dspy.ReAct` that researches via tools before generating output
- How ReAct manages an agentic loop, respects `max_iters`, and records its trajectory

## 7. Composing your own module

- How to build custom modules by subclassing `dspy.Module` and implementing `__init__` and `forward`
- How to compose submodules (e.g., `ReAct` + `ChainOfThought`) into multi-step pipelines
- How to route specific submodule calls to a different LM using `dspy.context`
- Why decomposition helps isolate context, reuse components, route work to cheaper models, and enable auditing

## 8. Building metrics for evaluation & optimization

- Why optimizers need metrics and common patterns for defining them (labeled data, rule-based checks, LLM judges)
- How to prepare `dspy.Example` objects, mark inputs with `.with_inputs`, and split data into train/val/test sets
- How to build a metric function and run a baseline evaluation with `dspy.Evaluate`

## 9. Prompt Optimizing with GEPA

- Why automated prompt optimization outperforms hand-tuning and transfers across models
- How GEPA uses a reflection LM and metric feedback (including text feedback) to iteratively rewrite instructions
- How to configure `dspy.GEPA` with a reflection LM, metric, and budget, then `compile` an optimized program
- How optimized prompts on smaller models can surpass unoptimized frontier models in quality, cost, and speed

## 10. Saving a program and reloading it

- The difference between saving state-only (`.json`, human-readable, needs reinstantiation) and saving the full program (directory with `save_program=True`, self-contained)
- How to reload programs using `dspy.load()` for full programs or `.load()` on a freshly instantiated module for state-only saves
- That saved programs do not include LM client configuration (API keys, provider, temperature), allowing the same program to target different models after loading