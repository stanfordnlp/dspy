---
sidebar_position: 1
---

# Learning DSPy: An Overview

DSPy exposes a very small number of APIs that you can learn quickly. However, building a compound AI system is a open-ended journey of
iterative development, in which you compose the tools and design patterns of DSPy to optimize for _your_ objectives. The three stages of
building AI systems in DSPy are:

1) **Building your DSPy program.** The first stage is defining your task, then choosing your language models, defining your signatures,
   and building your DSPy modules and combine them into a DSPy program.

2) **Evaluating your DSPy program.** Once your system starts working, this is the stage where you evalute how your program performs in
   your selected task. You need to collect a development/test dataset, and define your evaluation metric, the metric is an indicator
   of your program's performance.

3) **Optimizing your DSPy program.** Once you have a way to evaluate your system, you can use DSPy optimizers to improve your program. DSPy
   optimizers can tune both the prompts and weights in your program.

We suggest learning and applying DSPy in this order. For example, it's unproductive to launch optimization runs using a poorly-design program
or a bad metric.
