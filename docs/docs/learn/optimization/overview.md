---
sidebar_position: 1
---


# Optimization in DSPy

Once you have a system and a way to evaluate it, you can use DSPy optimizers to tune the prompts or weights in your program. Now it's useful to expand your data collection effort into building a training set and a held-out test set, in addition to the development set you've been using for exploration. For the training set (and its subset, validation set), you can often get substantial value out of 30 examples, but aim for at least 300 examples. Some optimizers accept a `trainset` only. Others ask for a `trainset` and a `valset`. When splitting data for most prompt optimizers, we recommend an unusual split compared to deep neural networks: 20% for training, 80% for validation. This reverse allocation emphasizes stable validation, since prompt-based optimizers often overfit to small training sets. In contrast, the [dspy.GEPA](https://dspy.ai/tutorials/gepa_ai_program/) optimizer follows the more standard ML convention: Maximize the training set size, while keeping the validation set just large enough to reflect the distribution of the downstream tasks (test set).

After your first few optimization runs, you are either very happy with everything or you've made a lot of progress but you don't like something about the final program or the metric. At this point, go back to step 1 (Programming in DSPy) and revisit the major questions. Did you define your task well? Do you need to collect (or find online) more data for your problem? Do you want to update your metric? And do you want to use a more sophisticated optimizer? Do you need to consider advanced features like DSPy Assertions? Or, perhaps most importantly, do you want to add some more complexity or steps in your DSPy program itself? Do you want to use multiple optimizers in a sequence?

Iterative development is key. DSPy gives you the pieces to do that incrementally: iterating on your data, your program structure, your metric, and your optimization steps. Optimizing complex LM programs is an entirely new paradigm that only exists in DSPy at the time of writing (update: there are now numerous DSPy extension frameworks, so this part is no longer true :-), so naturally the norms around what to do are still emerging. If you need help, we recently created a [Discord server](https://discord.gg/XCGy2WDCQB) for the community.

