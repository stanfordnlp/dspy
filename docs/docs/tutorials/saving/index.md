# Tutorial: Saving and Loading your DSPy program

This guide demonstrates how to save and load your DSPy program. At a high level, there are two ways to save your DSPy program:

1. Save the state of the program only, similar to weights-only saving in PyTorch.
2. Save the whole program, including both the architecture and the state, which is supported by `dspy>=2.6.0`.

## State-only Saving

State represents the DSPy program's internal state, including the signature, demos (few-shot examples), and other informaiton like
the `lm` to use for each `dspy.Predict` in the program. It also includes configurable attributes of other DSPy modules like
`k` for `dspy.retrievers.Retriever`. To save the state of a program, use the `save` method and set `save_program=False`. You can
choose to save the state to a JSON file or a pickle file. We recommend saving the state to a JSON file because it is safer and readable.
But sometimes your program contains non-serializable objects like `dspy.Image` or `datetime.datetime`, in which case you should save
the state to a pickle file.

Let's say we have compiled a program with some data, and we want to save the program for future usage:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)
```

To save the state of your program to json file:

```python
compiled_dspy_program.save("./dspy_program/model.json", save_program=False)
```

To save the state of your program to a pickle file:

```python
compiled_dspy_program.save("./dspy_program/model.pkl", save_program=False)
```

To load your saved state, you need to **recreate the same program**, then load the state using the `load` method.

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/model.json")
assert loaded_dspy_program.demos == compiled_dspy_program.demos
assert loaded_dspy_program.signature == compiled_dspy_program.signature
```

Or load the state from a pickle file:

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/model.pkl")
assert loaded_dspy_program.demos == compiled_dspy_program.demos
assert loaded_dspy_program.signature == compiled_dspy_program.signature
```

## Whole Program Saving

Starting from `dspy>=2.6.0`, DSPy supports saving the whole program, including the architecture and the state. This feature
is powered by `cloudpickle`, which is a library for serializing and deserializing Python objects.

To save the whole program, use the `save` method and set `save_program=True`, and specify a directory path to save the program
instead of a file name. We require a directory path because we also save some metadata, e.g., the dependency versions along
with the program itself.

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

To load the saved program, directly use `dspy.load` method:

```python
loaded_dspy_program = dspy.load("./dspy_program/")
assert loaded_dspy_program.demos == compiled_dspy_program.demos
assert loaded_dspy_program.signature == compiled_dspy_program.signature
```

With whole program saving, you don't need to recreate the program, but can directly load the architecture along with the state.
You can pick the suitable saviing approach based on your needs.

## Backward Compatibility

As of `dspy<2.7`, we don't guarantee the backward compatibility of the saved program. For example, if you save the program with `dspy==2.5.35`,
at loading time please make sure to use the same version of DSPy to load the program, otherwise the program may not work as expected. Chances
are that loading a saved file in a different version of DSPy will not raise an error, but the performance could be different from when
the program was saved.

Starting from `dspy>=2.7`, we will guarantee the backward compatibility of the saved program in major releases, i.e., programs saved in `dspy==2.7.0`
should be loadeable in `dspy==2.7.10`.
