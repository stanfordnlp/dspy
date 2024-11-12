---
sidebar_position: 1
---

# Learning DSPy: An Overview

As a framework, DSPy exposes a very small API that you can learn quickly. However, using it for novel tasks involves the more open-ended journey of composing its tools and design patterns to build _your_ modular AI systems quickly and to optimize them for _your_ objectives. This can be done along three different levels of system development, presented below.


1) **Programming in DSPy.** This is the stage in which your concern is to define your task, its constraints, explore a few examples of it, and use that to inform your initial pipeline design.

2) **Evaluation in DSPy.** Once you have a system that works reasonably well, you collect an initial small development set, define your DSPy metric, and use these to iterate on your system more systematically.

3) **Optimization in DSPy.** Once you have a system and a way to evaluate it, you can start to use DSPy optimizers to tune the prompts or weights in your program.

We typically suggest that you learn and apply DSPy in this order. For example, it's unproductive to launch optimization runs using a poorly-design program or a bad metric.