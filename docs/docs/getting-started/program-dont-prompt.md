# Program, don’t prompt

DSPy is a declarative way to build with LLMs. 

We describe our tasks as structured inputs and outputs. Often, we compose multiple tasks into multi-step programs and agents where each piece stays independently inspectable, swappable, and tunable. DSPy handles prompt construction, context management, and optimization that tunes each step to improve the whole program.

When a step needs to act on the world (to search, fetch, or compute), we give it tools as plain Python functions. Functions also serve as metrics: you define one to grade the output, and DSPy generates a better program accordingly. The optimizer tunes the prompt for any model you pick, so a small, cheap model can often match or beat a hand-prompted frontier one..

DSPy helps us program LLMs, rather than prompting them, creating modular, maintainable and optimizable AI software.

## What we’ll learn today

In this tutorial we’ll build a haiku-writing program that starts with four lines of Python and grows into a tool-using, prompt-optimized agent. Along the way we’ll touch each of DSPy’s core components. We’ll learn:

- How to install DSPy, configure a **language model** and write a simple DSPy program.  
- What a **Signature** is, and why DSPy uses signatures instead of hand-written prompt strings.  
- What a **Module** is, how `Predict`, `ChainOfThought`, and `ReAct` differ, and when to reach for each.  
- How to compose a custom `dspy.Module` that decomposes a task into named, independent stages.  
- How to write **metrics** and use **optimizers** to compile better versions of our program.  
- How to save optimized programs and reload them.

---

**Next:** [Setting up DSPy →](installation.md)
