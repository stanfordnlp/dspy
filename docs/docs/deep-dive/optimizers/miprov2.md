# `MIPROv2` Optimizer

## Table of Contents
1. [Overview](#overview)
2. [Example Usage](#example-usage)
   - [Setting up a Sample Pipeline](#setting-up-a-sample-pipeline)
   - [Optimizing with MIPROv2](#optimizing-with-miprov2)
   - [Optimizing instructions only with MIPROv2 (0-Shot)](#optimizing-instructions-only-with-miprov2-0-shot)
3. [Parameters](#parameters)
   - [Initialization Parameters](#initialization-parameters)
   - [Compile Method Specific Parameters](#compile-method-specific-parameters)
4. [How MIPROv2 Works](#how-miprov2-works)

## Overview

`MIPROv2` (<u>M</u>ultiprompt <u>I</u>nstruction <u>PR</u>oposal <u>O</u>ptimizer Version 2) is an prompt optimizer capable of optimizing both instructions and few-shot examples jointly. It does this by bootstrapping few-shot example candidates, proposing instructions grounded in different dynamics of the task, and finding an optimized combination of these options using Bayesian Optimization. It can be used for optimizing few-shot examples & instructions jointly, or just instructions for 0-shot optimization.

## Example Usage

### Setting up a Sample Pipeline

We'll be making a basic answer generation pipeline over GSM8K dataset that we saw in the [Minimal Example](https://dspy-docs.vercel.app/docs/quick-start/minimal-example), we won't be changing anything in it! So let's start by configuring the LM which will be OpenAI LM client with `gpt-3.5-turbo` as the LLM in use.

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=250)
dspy.settings.configure(lm=turbo)
```

Now that we have the LM client setup it's time to import the train-dev split in `GSM8k` class that DSPy provides us:

```python
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

gms8k = GSM8K()

trainset, devset = gms8k.train, gms8k.dev
```

We'll now define a basic QA inline signature i.e. `question->answer` and pass it to `ChainOfThought` module, that applies necessary addition for CoT style prompting to the Signature.

```python
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
```

Now we need to evaluate this pipeline too! So we'll use the `Evaluate` class that DSPy provides us, as for the metric we'll use the `gsm8k_metric` that we imported above.

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=8, display_progress=True, display_table=False)
```

To evaluate the `CoT` pipeline we'll need to create an object of it and pass it as an arg to the `evaluator` call.

```python
program = CoT()

evaluate(program, devset=devset[:])
```

Now we have the baseline pipeline ready to use, so let's try using the `MIPROv2` optimizer to improve our pipeline's performance!

### Optimizing with `MIPROv2`
Here we show how to import, initialize, and compile our program with optimized few-shot examples and instructions using `MIPROv2`.

```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    num_candidates=7,
    init_temperature=0.5,
    verbose=False,
    num_threads=4,
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    num_trials=15,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    minibatch=True, 
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

### Optimizing instructions only with `MIPROv2` (0-Shot)

In some cases, we may want to only optimize the instruction, rather than including few-shot examples in the prompt. The code below demonstrates how this can be done using `MIPROv2`. Note that the key difference involves setting `max_labeled_demos` and `max_bootstrapped_demos` to zero.
```python
# Import optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer 
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    num_candidates=7,
    init_temperature=0.5,
    verbose=False,
    num_threads=4,
)

# Perform optimization
print(f"Optimizing program with MIPRO (0-Shot)...")
zeroshot_optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0, # setting demos to 0 for 0-shot optimization
    max_labeled_demos=0,
    num_trials=15,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    minibatch=False, 
    requires_permission_to_run=False,
)


zeroshot_optimized_program.save(f"mipro_0shot_optimized")

print(f"Evaluate optimized program...")
evaluate(zeroshot_optimized_program, devset=devset[:])
```

## Parameters

### Initialization Parameters

| Parameter            | Type         | Default                                              | Description                                                                                                                  |
|----------------------|--------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `metric`           | `dspy.metric`       | N/A - Required                                                 | The evaluation metric used to optimize the task model.                                                                       |
| `prompt_model`     | `dspy.LM`       | LM specified in `dspy.settings`                        | Model used for prompt generation.                                                                                            |
| `task_model`       | `dspy.LM`       | LM specified in `dspy.settings`                        | Model used for task execution.                                                                                               |
| `num_candidates`   | `int`      | `10`                                                   | Number of candidate instructions & few-shot examples to generate and evaluate for each predictor. If `num_candidates=10`, this means for a 2 module LM program we'll be optimizing over 10 candidates x 2 modules x 2 variables (few-shot ex. and instructions for each module)= 40 total variables. Therfore, if we increase `num_candidates`, we will probably want to increase `num_trials` as well (see Compile parameters).                                                                          |
| `num_threads`      | `int`      |  `6`                                       | Threads to use for evaluation.                                                                                               |
| `max_errors`       | `int`      | `10`                                                 | Maximum errors during an evaluation run that can be made before throwing an Exception.                                       |
| `teacher_settings` | `dict`       | `{}`                                                 | Settings to use for the teacher model that bootstraps few-shot examples. An example dict would be `{lm=<dspy.LM object>}`. If your LM program with your default model is struggling to bootstrap any examples, it could be worth using a more powerful teacher model for bootstrapping.                                                     |
| `init_temperature` | `float`        | `1.0`                                                  | The initial temperature for prompt generation, influencing creativity.                                                       |
| `verbose`          | `bool`      | `False`                                                | Enables printing intermediate steps and information.                                                                         |
| `track_stats`      | `bool`      | `True`                                                | Logs relevant information through the optimization process if set to True.                                                                       |
| `metric_threshold` | `float`        | `None`                                                 | A metric threshold is used if we only want to keep bootstrapped few-shot examples that exceed some threshold of performance.  |

### Compile Parameters

| Parameter                  | Type     | Default | Description                                                                                              |
|----------------------------|----------|---------|----------------------------------------------------------------------------------------------------------|
| `student`                | `dspy.Module`   | N/A (Required)    | The base program to optimize.                                                                            |
| `trainset`               | `List[dspy.Example]`  | N/A (Required)    | Training dataset which is used to bootstrap few-shot examples and instructions. If a separate `valset` is not specified, 80% of this training set will also be used as a validation set for evaluating new candidate prompts.                                                                    |
| `valset`               | `List[dspy.Example]`  | Defaults to 80% of trainset | Dataset which is used to evaluate candidate prompts. We recommend using somewhere between 50-500 examples for optimization.                                                                      |
| `num_trials`            | `int`  | `30`      | Number of optimization trials to run. When `minibatch` is set to `True`, this represents the number of minibatch trials that will be run on batches of size `minibatch_size`. When minibatch is set to `False`, each trial uses a full evaluation on the training set. In both cases, we recommend setting `num_trials` to a *minimum* of .75 x # modules in program x # variables per module (2 if few-shot examples & instructions will both be optimized, 1 in the 0-shot case).        |
| `minibatch`              | `bool`  | `True`    | Flag to enable evaluating over minibatches of data (instead of the full validation set) for evaluation each trial.                                                           |
| `minibatch_size`         | `int`  | `25.0`    | Size of minibatches for evaluations.                                                                     |
| `minibatch_full_eval_steps` | `int` | `10`    | When minibatching is enabled, a full evaluation on the validation set will be carried out every `minibatch_full_eval_steps` on the top averaging set of prompts (according to their average score on the minibatch trials).          
| `max_bootstrapped_demos` | `int`  | `4`      | Maximum number of bootstrapped demonstrations to generate and include in the prompt.                              |
| `max_labeled_demos`      | `int`  | `16`      | Maximum number of labeled demonstrations to generate and include in the prompt. Note that these differ from bootstrapped examples because they are just inputs & outputs sampled directly from the training set and do not have bootstrapped intermediate steps.                             |
| `seed`                   | `int`  | `9`       | Seed for reproducibility.                                                    |                                  |
| `program_aware_proposer` | `bool`  | `True`    | Flag to enable summarizing a reflexive view of the code for your LM program.                             |
| `data_aware_proposer` | `bool`  | `True`    | Flag to enable summarizing your training dataset.                             |
| `view_data_batch_size` | `int`  | `10`    | Number of data examples to look at a time when generating the summary.                             |
| `tip_aware_proposer` | `bool`  | `True`    | Flag to enable using a randomly selected tip for instruction generation.                             |
| `fewshot_aware_proposer` | `bool`  | `True`    | Flag to enable using generated few-shot examples for instruction proposal.                             |
| `requires_permission_to_run` | `bool` | `True`  | Flag to require user confirmation before running optimizations.                                          |

## How `MIPROv2` works

At a high level, `MIPROv2` works by creating both few-shot examples and new instructions for each predictor in your LM program, and then searching over these using Bayesian Optimization to find the best combination of these variables for your program.

These steps are broken down in more detail below:
1) **Bootstrap Few-Shot Examples**: The same bootstrapping technique used in `BootstrapFewshotWithRandomSearch` is used to create few-shot examples. This works by randomly sampling examples from your training set, which are then run through your LM program. If the output from the program is correct for this example, it is kept as a valid few-shot example candidate. Otherwise, we try another example until we've curated the specified amount of few-shot example candidates. This step creates `num_candidates` sets of `max_bootstrapped_demos` bootstrapped examples and `max_labeled_demos` basic examples sampled from the training set.
2) **Propose Instruction Candidates**. Next, we propose instruction candidates for each predictor in the program. This is done using another LM program as a proposer, which bootstraps & summarizes relevant information about the task to generate high quality instructions. Specifically, the instruction proposer includes (1) a generated summary of properties of the training dataset, (2) a generated summary of your LM program's code and the specific predictor that an instruction is being generated for, (3) the previously bootstrapped few-shot examples to show reference inputs / outputs for a given predictor and (4) a randomly sampled tip for generation (i.e. "be creative", "be concise", etc.) to help explore the feature space of potential instructions.
3. **Find an Optimized Combination of Few-Shot Examples & Instructions**. Finally, now that we've created these few-shot examples and instructions, we use Bayesian Optimization to choose which set of these would work best for each predictor in our program. This works by running a series of `num_batches` trials, where a new set of prompts are evaluated over our validation set at each trial. This helps the Bayesian Optimizer learn which combination of prompts work best over time. If `minibatch` is set to `True` (which it is by default), then the new set of prompts are only evaluated on a minibatch of size `minibatch_size` at each trial which generally allows for more efficient exploration / exploitation. The best averaging set of prompts is then evalauted on the full validation set every `minibatch_full_eval_steps` get a less noisey performance benchmark. At the end of the optimization process, the LM program with the set of prompts that performed best on the full validation set is returned.


For those interested in more details, more information on `MIPROv2` along with a study on `MIPROv2` compared with other DSPy optimizers can be found in [this paper](https://arxiv.org/abs/2406.11695). 