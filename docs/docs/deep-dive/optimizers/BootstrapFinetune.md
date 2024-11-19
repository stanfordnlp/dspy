---
sidebar_position: 4
---

# teleprompt.BootstrapFinetune

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


### Constructor

### `__init__(self, metric=None, teacher_settings={}, multitask=True)`

The constructor initializes a `BootstrapFinetune` instance and sets up its attributes. It defines the teleprompter as a `BootstrapFewShot` instance for the finetuning compilation.

```python
class BootstrapFinetune(Teleprompter):
    def __init__(self, metric=None, teacher_settings={}, multitask=True):
```

**Parameters:**
- `metric` (_callable_, _optional_): Metric function to evaluate examples during bootstrapping. Defaults to `None`.
- `teacher_settings` (_dict_, _optional_): Settings for teacher predictor. Defaults to empty dictionary.
- `multitask` (_bool_, _optional_): Enable multitask fine-tuning. Defaults to `True`.

### Method

#### `compile(self, student, *, teacher=None, trainset, valset=None, target='t5-large', bsize=12, accumsteps=1, lr=5e-5, epochs=1, bf16=False)`

This method first compiles for bootstrapping with the `BootstrapFewShot` teleprompter. It then prepares fine-tuning data by generating prompt-completion pairs for training and performs finetuning. After compilation, the LMs are set to the finetuned models and the method returns a compiled and fine-tuned predictor.

**Parameters:**
- `student` (_Predict_): Student predictor to be fine-tuned.
- `teacher` (_Predict_, _optional_): Teacher predictor to help with fine-tuning. Defaults to `None`.
- `trainset` (_list_): Training dataset for fine-tuning.
- `valset` (_list_, _optional_): Validation dataset for fine-tuning. Defaults to `None`.
- `target` (_str_, _optional_): Target model for fine-tuning. Defaults to `'t5-large'`.
- `bsize` (_int_, _optional_): Batch size for training. Defaults to `12`.
- `accumsteps` (_int_, _optional_): Gradient accumulation steps. Defaults to `1`.
- `lr` (_float_, _optional_): Learning rate for fine-tuning. Defaults to `5e-5`.
- `epochs` (_int_, _optional_): Number of training epochs. Defaults to `1`.
- `bf16` (_bool_, _optional_): Enable mixed-precision training with BF16. Defaults to `False`.

**Returns:**
- `compiled2` (_Predict_): A compiled and fine-tuned `Predict` instance.

### Example

```python
#Assume defined trainset
#Assume defined RAG class
...

#Define teleprompter
teleprompter = BootstrapFinetune(teacher_settings=dict({'lm': teacher}))

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset, target='google/flan-t5-base')
```