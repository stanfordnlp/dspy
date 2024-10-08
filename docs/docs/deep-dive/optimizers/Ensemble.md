---
sidebar_position: 4
---

# teleprompt.Ensemble

### Constructor

The constructor initializes the `Ensemble` class and sets up its attributes. This teleprompter is designed to create ensembled versions of multiple programs, reducing various outputs from different programs into a single output.

```python
class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
```

**Parameters:**
- `reduce_fn` (_callable_, _optional_): Function used to reduce multiple outputs from different programs into a single output. A common choice is `dspy.majority`. Defaults to `None`.
- `size` (_int_, _optional_): Number of programs to randomly select for ensembling. If not specified, all programs will be used. Defaults to `None`.
- `deterministic` (_bool_, _optional_): Specifies whether ensemble should operate deterministically. Currently, setting this to `True` will raise an error as this feature is pending implementation. Defaults to `False`.

### Method

#### `compile(self, programs)`

This method compiles an ensemble of programs into a single program that when run, can either randomly sample a subset of the given programs to produce outputs or use all of them. The multiple outputs can then be reduced into a single output using the `reduce_fn`.

**Parameters:**
- `programs` (_list_): List of programs to be ensembled.

**Returns:**
- `EnsembledProgram` (_Module_): An ensembled version of the input programs.

### Example

```python
import dspy
from dspy.teleprompt import Ensemble

# Assume a list of programs
programs = [program1, program2, program3, ...]

# Define Ensemble teleprompter
teleprompter = Ensemble(reduce_fn=dspy.majority, size=2)

# Compile to get the EnsembledProgram
ensembled_program = teleprompter.compile(programs)
```