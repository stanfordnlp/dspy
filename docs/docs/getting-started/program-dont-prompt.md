# Program, don’t prompt

A capable language model is not yet a system. A system has behavior we can repeat, inspect, change, evaluate, and improve. **DSPy is a Python framework for building those systems.**

You do not need to become a prompt engineer—or an ML expert—to begin. Start by describing one useful transformation as named inputs and outputs. DSPy turns that declaration into the messages a model needs. As the application grows, ordinary Python gives it structure, and examples and metrics tell DSPy what to improve.

The goal is intelligence you can program, not an oracle you prompt and hope.

## Express intent where it fits

Most AI systems need three forms of intent:

1. **Python for what must happen exactly.** Use normal functions, branches, loops, permissions, and state for the parts that should behave like normal software.
2. **Signatures for what requires judgment.** Describe a fuzzy transformation with typed, meaningful inputs and outputs—for example, `customer_message -> intent, urgency`.
3. **Examples and metrics for what good looks like.** Use data, checks, or judgments to capture variation and the difficult cases that are easier to recognize than enumerate.

You rarely need all three at first. A four-line program with one signature is a good start. Add explicit control flow when the workflow needs it. Add data and optimization once you can say how to distinguish better results from worse ones.

This division matters. If a rule must always hold, write it in Python or a type instead of asking the model to remember it. If a task genuinely needs interpretation, isolate that ambiguity in a signature instead of hard-coding a brittle approximation. If you already know a requirement, state it; optimization is for adapting to models and handling varied edge cases, not for guessing intent you could have expressed directly.

## A small programming model

DSPy keeps the task you mean separate from the techniques used to perform it:

- A **Signature** declares the transformation: its named, typed inputs and outputs.
- A **Module** chooses an inference strategy. The same signature can run with `Predict`, `ChainOfThought`, `ReAct`, or a module you compose in plain Python.
- A **metric** scores behavior against your objective.
- An **optimizer** uses that feedback to compile a better version of the whole program—by improving instructions, selecting demonstrations, tuning weights, or using newer methods as they emerge.

That separation makes each piece independently inspectable, swappable, and tunable. A better model, inference strategy, or optimizer should improve your system without forcing you to rewrite what the system means.

DSPy handles prompt construction, structured parsing, context management, and optimization. You keep ownership of the intent and the software around it.

## What we’ll build

In this tutorial, we’ll start with a four-line haiku writer and grow it into a tool-using, optimized program. Along the way we’ll learn:

- how to install DSPy, configure a **language model**, and write a simple program;
- how **Signatures** express tasks without model-specific prompt strings;
- how `Predict`, `ChainOfThought`, and `ReAct` provide different **Module** strategies for the same task;
- how to compose a custom module with normal Python control flow;
- how tools connect a program to search, retrieval, and computation;
- how **metrics** express success and **optimizers** compile better programs;
- how to inspect, save, and reload the resulting program.

These ideas also guide [what belongs in DSPy and where the project is going](../roadmap.md).

---

**Next:** [Setting up DSPy →](installation.md)
