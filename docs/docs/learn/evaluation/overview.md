---
sidebar_position: 1
---

# Evaluation in DSPy

Once you have a system that works reasonably well, it's time to **collect an initial development set**. This will help you refine your system more systematically. Even 20 input examples of your task can be useful, though 200 goes a long way. You could prepare a few examples by hand: depending on your _metric_, you either just need inputs and no labels at all, or you need inputs and the _final_ outputs of your system. (You almost never need labels for the intermediate steps in your program in DSPy.) Alternatively, you can probably find datasets that are adjacent to your task on, say, HuggingFace datasets or in a naturally occuring source like StackExchange. If there's data whose licenses are permissive enough, we suggest you use them. Otherwise, you can also start deploying a demo of your system and collect initial data that way.

Next, you should **define your DSPy metric**. What makes outputs from your system good or bad? Invest in defining metrics and improving them incrementally over time. It's really hard to consistently improve what you aren't able to define. A metric is just a function that will take examples from your data and take the output of your system, and return a score that quantifies how good the output is. For simple tasks, this could be just "accuracy" or "exact match" or "F1 score". This may be the case for simple classification or short-form QA tasks. For most applications, your system will produce long-form outputs, so your metric will be a smaller DSPy program that checks multiple properties of the output. Getting this right on the first try is unlikely, but you should start with something simple and iterate.

Now that you have some data and a metric, run development evaluations on your pipeline designs to understand their tradeoffs. Look at the outputs and the metric scores. This will probably allow you to spot any major issues, and it will define a baseline for your next steps.


??? "If your metric is itself a DSPy program..."
    If your metric is itself a DSPy program, a powerful way to iterate is to optimize your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5), so the metric's metric is easy to define and optimize by collecting a few examples.

