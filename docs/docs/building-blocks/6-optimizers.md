---
sidebar_position: 6
---

# Optimizers (formerly Teleprompters)

A **DSPy optimizer** is an algorithm that can tune the parameters of a DSPy program (i.e., the prompts and/or the LM weights) to maximize the metrics you specify, like accuracy.

There are many built-in optimizers in DSPy, which apply vastly different strategies. A typical DSPy optimizer takes three things:

- Your **DSPy program**. This may be a single module (e.g., `dspy.Predict`) or a complex multi-module program.

- Your **metric**. This is a function that evaluates the output of your program, and assigns it a score (higher is better).

- A few **training inputs**. This may be very small (i.e., only 5 or 10 examples) and incomplete (only inputs to your program, without any labels).

If you happen to have a lot of data, DSPy can leverage that. But you can start small and get strong results.

**Note:** Formerly called **DSPy Teleprompters**. We are making an official name update, which will be reflected throughout the library and documentation.


## **What** does a DSPy Optimizer tune? **How** does it tune them?

Traditional deep neural networks (DNNs) can be optimized with gradient descent, given a loss function and some training data.

DSPy programs consist of multiple calls to LMs, stacked together as [DSPy modules]. Each DSPy module has internal parameters of three kinds: (1) the LM weights, (2) the instructions, and (3) demonstrations of the input/output behavior.

Given a metric, DSPy can optimize all of these three with multi-stage optimization algorithms. These can combine gradient descent (for LM weights) and discrete LM-driven optimization, i.e. for crafting/updating instructions and for creating/validating demonstrations. DSPy Demonstrations are like few-shot examples, but they're far more powerful. They can be created from scratch, given your program, and their creation and selection can be optimized in many effective ways.

In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things, much more systematically, and tune the metrics directly.


## What DSPy Optimizers are currently available?

All of these can be accessed via `from dspy.teleprompt import *`.

#### Automatic Few-Shot Learning

1. **`LabeledFewShot`**: Simply constructs few-shot examples from provided labeled Q/A pairs.

2. **`BootstrapFewShot`**: Uses your program to self-generate complete demonstrations for every stage of your program. Will simply use the generated demonstrations (if they pass the metric) without any further optimization. Advanced: Supports using a teacher program (a different DSPy program that has compatible structure) and a teacher LM, for harder tasks.

3. **`BootstrapFewShotWithRandomSearch`**: Applies `BootstrapFewShot` several times with random search over generated demonstrations, and selects the best program.

4. **`BootstrapFewShotWithOptuna`**: Applies `BootstrapFewShot` through Optuna hyperparameter optimization across demonstration sets, running trials to maximize evaluation metrics. 


#### Automatic Instruction Optimization

4. **`SignatureOptimizer`**: Generates and refines new instructions for each step, and optimizes them with coordinate ascent.

5. **`BayesianSignatureOptimizer`**: Generates instructions and few-shot examples in each step. The instruction generation is data-aware and demonstration-aware. Uses Bayesian Optimization to effectively search over the space of generation instructions/demonstrations across your modules.


#### Automatic Finetuning

6. **`BootstrapFinetune`**: Distills a prompt-based DSPy program into weight updates (for smaller LMs). The output is a DSPy program that has the same steps, but where each step is conducted by a finetuned model instead of a prompted LM.


#### Program Transformations

7. **`KNNFewShot`**. Selects demonstrations through k-Nearest Neighbors algorithm integrating `BootstrapFewShot` for bootstrapping/selection process. 

8. **`Ensemble`**: Ensembles a set of DSPy programs and either uses the full set or randomly samples a subset into a single program.


## Which optimizer should I use?

As a rule of thumb, if you don't know where to start, use `BootstrapFewShotWithRandomSearch`.

Here's the general guidance on getting started:

* If you have very little data, e.g. 10 examples of your task, use `BootstrapFewShot`.

* If you have slightly more data, e.g. 50 examples of your task, use `BootstrapFewShotWithRandomSearch`.

* If you have more data than that, e.g. 300 examples or more, use `BayesianSignatureOptimizer`.

* If you have been able to use one of these with a large LM (e.g., 7B parameters or above) and need a very efficient program, compile that down to a small LM with `BootstrapFinetune`.


## How do I use an optimizer?

They all share this general interface, with some differences in the keyword arguments (hyperparameters).

Let's see this with the most common one, `BootstrapFewShotWithRandomSearch`.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
# The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
config = dict(max_bootstrapped_demos=3, max_labeled_demos=3, num_candidate_programs=10, num_threads=4)

teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config)
optimized_program = teleprompter.compile(YOUR_PROGRAM_HERE, trainset=YOUR_TRAINSET_HERE)
```

## Saving and loading optimizer output

After running a program through an optimizer, it's useful to also save it. At a later point, a program can be loaded from a file and used for inference. For this, the `load` and `save` methods can be used.

### Saving a program

```python
optimized_program.save(YOUR_SAVE_PATH)
```

The resulting file is in plain-text JSON format. It contains all the parameters and steps in the source program. You can always read it and see what the optimizer generated.

### Loading a program

To load a program from a file, you can instantiate an object from that class and then call the load method on it.

```python
loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load(path=YOUR_SAVE_PATH)
```