---
sidebar_position: 1
---

# Programming in DSPy

DSPy is a bet on _writing code instead of strings_. In other words, building the right control flow is crucial. Start by **defining your task**. What are the inputs to your system and what should your system produce as output? Is it a chatbot over your data or perhaps a code assistant? Or maybe a system for translation, for highlighting snippets from search results, or for generating reports with citations?

Next, **define your initial pipeline**. Can your DSPy program just be a single module or do you need to break it down into a few steps? Do you need retrieval or other tools, like a calculator or a calendar API? Is there a typical workflow for solving your problem in multiple well-scoped steps, or do you want more open-ended tool use with agents for your task? Think about these but start simple, perhaps with just a single `dspy.ChainOfThought` module, then add complexity incrementally based on observations.

As you do this, **craft and try a handful of examples** of the inputs to your program. Consider using a powerful LM at this point, or a couple of different LMs, just to understand what's possible. Record interesting (both easy and hard) examples you try. This will be useful when you are doing evaluation and optimization later.


??? "Beyond encouraging good design patterns, how does DSPy help here?"

    Conventional prompts couple your fundamental system architecture with incidental choices not portable to new LMs, objectives, or pipelines. A conventional prompt asks the LM to take some inputs and produce some outputs of certain types (a _signature_), formats the inputs in certain ways and requests outputs in a form it can parse accurately (an _adapter_), asks the LM to apply certain strategies like "thinking step by step" or using tools (a _module_'s logic), and relies on substantial trial-and-error to discover the right way to ask each LM to do this (a form of manual _optimization_).
    
    DSPy separates these concerns and automates the lower-level ones until you need to consider them. This allow you to write much shorter code, with much higher portability. For example, if you write a program using DSPy modules, you can swap the LM or its adapter without changing the rest of your logic. Or you can exchange one _module_, like `dspy.ChainOfThought`, with another, like `dspy.ProgramOfThought`, without modifying your signatures. When you're ready to use optimizers, the same program can have its prompts optimized or its LM weights fine-tuned.
