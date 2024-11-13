---
sidebar_position: 1
---

# Learning DSPy: An Overview

DSPy exposes a very small API that you can learn quickly, but building new AI systems is a more open-ended journey of composing the tools and design patterns in DSPy to optimize for _your_ objectives. The three stages of building AI systems in DSPy are:

1) **DSPy Programming.** This is all about defining your task, its constraints, exploring a few examples, and using that to inform your initial pipeline design.

2) **DSPy Evaluation.** Once your system works reasonably well, this is the stage where you collect an initial development set, define your DSPy metric, and use these to iterate on your system more systematically.

3) **DSPy Optimization.** Once you have a way to evaluate your system, you use DSPy optimizers to tune the prompts or weights in your program.

We suggest learning and applying DSPy in this order. For example, it's unproductive to launch optimization runs using a poorly-design program or a bad metric.