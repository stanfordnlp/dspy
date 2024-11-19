---
sidebar_position: 3
---

# BootstrapFewShotWithRandomSearch

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


`BootstrapFewShotWithRandomSearch` is an teleprompter that extends `BootstrapFewShot` by incorporating a random search strategy to optimize the selection of few-shot examples. This teleprompter is useful when you want to explore a variety of example combinations to find the best examples for your model.

## Usage

In terms of API `BootstrapFewShotWithRandomSearch` teleprompter is quite similar to `BootstrapFewShot` with few additional parameters needed. More precisely the parameters are:

- `metric` (callable): The metric function used to evaluate examples during bootstrapping and final evaluation.
- `teacher_settings` (dict, optional): Settings for the teacher predictor. Defaults to an empty dictionary.
- `max_bootstrapped_demos` (int, optional): Maximum number of bootstrapped demonstrations per predictor. Defaults to 4.
- `max_labeled_demos` (int, optional): Maximum number of labeled demonstrations per predictor. Defaults to 16.
- `max_rounds` (int, optional): Maximum number of bootstrapping rounds. Defaults to 1.
- `num_candidate_programs` (int): Number of candidate programs to generate during random search. Defaults to 16.
- `num_threads` (int): Number of threads used for evaluation during random search. Defaults to 6.
- `max_errors` (int): Maximum errors permitted during evaluation. Halts run with the latest error message. Defaults to 10.
- `stop_at_score` (float, optional): Score threshold for random search to stop early. Defaults to None.
- `metric_threshold` (float, optional): Score threshold for the metric to determine a successful example. Defaults to None.

## Working Example

Let's take the example of optimizing a simple CoT pipeline for GSM8k dataset, we'll take the example in [BootstrapFewShot](/deep-dive/optimizers/bootstrap-fewshot) as our running example for optimizers. We're gonna assume our data and pipeline is same as the on in `BootstrapFewShot` article. So let's start by initializing the optimizer:

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

teleprompter = BootstrapFewShotWithRandomSearch(
    metric=gsm8k_metric, 
    max_bootstrapped_demos=8, 
    max_labeled_demos=8,
    num_threads=10,
    num_candidate_programs=10
)
```

Now that we have initialized our teleprompter, let's compile our CoT module with the call to `compile` standard to all optimizers:

```python
cot_compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)
```

Once the training is done you'll have a more optimized module that you can save or load again for use anytime:

```python
cot_compiled.save('turbo_gsm8k.json')

# Loading:
# cot = CoT()
# cot.load('turbo_gsm8k.json')
```