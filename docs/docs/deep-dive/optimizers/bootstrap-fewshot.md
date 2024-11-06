---
sidebar_position: 1
---

# BootstrapFewShot

When compiling a DSPy program, we generally invoke an optimizer that takes the program, a training set, and a metric and returns a new optimized program. Different optimizers apply different strategies for optimization. This family of optimizers is focused on optimizing the few shot examples. Let's take an example of a Sample pipeline and see how we can use this optimizer to optimize it.

## Setting up a Sample Pipeline

We will be making a basic answer generation pipeline over the GSM8K dataset. We won't be changing anything in it! So let's start by configuring the LM which will be OpenAI LM client with `gpt-3.5-turbo` as the LLM in use.

```python
import dspy

turbo = dspy.LM(model='openai/gpt-3.5-turbo', max_tokens=250)
dspy.configure(lm=turbo)
```

Now that we have the LM client setup it's time to import the train-dev split in `GSM8k` class that DSPy provides us:

```python
import random
from dspy.datasets import DataLoader

dl = DataLoader()

gsm8k = dl.from_huggingface(
    "gsm8k",
    "main",
    input_keys = ("question",),
)

trainset, devset = gsm8k['train'], random.sample(gsm8k['test'], 50)
```

We will now define a basic QA inline signature i.e. `question->answer` and pass it to `ChainOfThought` module, that applies necessary addition for CoT style prompting to the Signature.

```python
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)
```

Now we need to evaluate this pipeline too!! So we'll use the `Evaluate` class that DSPy provides us, as for the metric we'll use the `gsm8k_metric` that we imported above.

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=NUM_THREADS, display_progress=True, display_table=False)
```

To evaluate the `CoT` pipeline we'll need to create an object of it and pass it as an arg to the `evaluator` call.

```python
cot_baseline = CoT()

evaluate(cot_baseline)
```

Now we have the baseline pipeline ready to use, so let's try using the `BootstrapFewShot` optimizer and optimizing our pipeline to make it even better!

## Using `BootstrapFewShot`

Let's start by importing and initializing our optimizer, for the metric we'll be using the same `gsm8k_metric` imported and used above:

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=gsm8k_metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
    max_rounds=10,
)
```

`metric` refers to the metric that we want to optimize the pipeline for.

`max_rounds` refers to the maximum number of rounds of training we want to perform. During the bootstrapping process we train the student on the teacher's predictions and then use the teacher to generate new examples to add to the training set. This process is repeated for `max_rounds` times and for each round, the `temperature` parameter of the teacher is increased from a baseline value of `0.7` to `0.7 + 0.001 * round_number`.

What are `max_bootstrapped_demos` and `max_labeled_demos` parameters? Let's see the differences via a table:

| Feature                   | `max_labeled_demos`                                                                                                                                                                                                                        | `max_bootstrapped_demos`                                                                                                                                                                                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Refers to the maximum number of labeled demonstrations (examples) that will be used for training the student module directly. Labeled demonstrations are typically pre-existing, manually labeled examples that the module can learn from. | Refers to the maximum number of demonstrations that will be bootstrapped. Bootstrapping in this context means generating new training examples based on the predictions of the teacher module or some other process. These bootstrapped demonstrations are then used alongside or instead of the manually labeled examples. |
| **Training Usage**        | Directly used in training; typically more reliable due to manual labeling.                                                                                                                                                                 | Augment training data; potentially less accurate as they are generated examples.                                                                                                                                                                                                                                            |
| **Data Source**           | Pre-existing dataset of manually labeled examples.                                                                                                                                                                                         | Generated during the training process, often using outputs from a teacher module.                                                                                                                                                                                                                                           |
| **Influence on Training** | Higher quality and reliability, assuming labels are accurate.                                                                                                                                                                              | Provides more data but may introduce noise or inaccuracies.                                                                                                                                                                                                                                                                 |

This optimizer augments any necessary field even if you data doesn't have it, for example we don't have rationale for labelling however you'll see the rationale for each few shot example in the prompt that this optimizer curated, how? By generating them all via a `teacher` module which is an optional parameter, since we didn't pass it the optimizer creates a teacher from the module we are training or the `student` module.

In the next section, we'll seen this process step by step but for now let's start optimizing our `CoT` module by calling the `compile` method in the optimizer:

```python
cot_compiled = optimizer.compile(CoT(), trainset=trainset, valset=devset)
```

Once the training is done you'll have a more optimized module that you can save or load again for use anytime:

```python
cot_compiled.save('turbo_gsm8k.json')

# Loading:
# cot = CoT()
# cot.load('turbo_gsm8k.json')
```

## How `BootstrapFewShot` works?

`LabeledFewShot` is the most vanilla optimizer that takes a training set as input and it assigns subsets of the trainset in each student's predictor's demos attribute. You can think of it as basically adding few shot examples to the prompt.

`BootStrapFewShot` follows the following steps:

1. Initializing a student program which is the one we are optimizing and a teacher program which unless specified otherwise is a clone of the student.

2. Using the `LabeledFewShot` optimizer it adds the labeled demonstrations to the teacher's predictor's `demos` attribute. The `LabeledFewShot` optimizer randomly samples `max_labeled_demos` from the trainset.

3. Mappings are created between the names of predictors and their corresponding instances in both student and teacher models.

4. Then it bootstraps the teacher by generating `max_bootstrapped_demos` examples using the inputs from the trainset and the teacher's predictions.

5. The process iterates over each example in the training set. For each example, the method checks if the maximum number of bootstraps has been reached. If so, the process stops.

6. For each training example, the teacher model attempts to generate a prediction.

7. If the teacher model successfully generates a prediction that satisfies the metric, the trace of this prediction process is captured. This trace includes details about which predictors were called, the inputs they received, and the outputs they produced.

8. If the prediction is successful, a demonstration (demo) is created for each step in the trace. This demo includes the inputs to the predictor and the outputs it generated.

9. Other optimizers like `BootstrapFewShotWithOptuna`, `BootstrapFewShotWithRandomSearch` etc. also work on same principles with slight changes in the example discovery process.

