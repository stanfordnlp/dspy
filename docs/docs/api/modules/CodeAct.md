# dspy.CodeAct

<!-- START_API_REF -->
::: dspy.CodeAct
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - get_lm
            - inspect_history
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->

# CodeAct

CodeAct is a DSPy module that combines code generation with tool execution to solve problems. It generates Python code snippets that use provided tools and the Python standard library to accomplish tasks.

## Basic Usage

Here's a simple example of using CodeAct:

```python
import dspy
from dspy.predict import CodeAct

# Define a simple tool function
def factorial(n: int) -> int:
    """Calculate the factorial of a number."""
    if n == 1:
        return 1
    return n * factorial(n-1)

# Create a CodeAct instance
act = CodeAct("n->factorial_result", tools=[factorial])

# Use the CodeAct instance
result = act(n=5)
print(result) # Will calculate factorial(5) = 120
```

## How It Works

CodeAct operates in an iterative manner:

1. Takes input parameters and available tools
2. Generates Python code snippets that use these tools
3. Executes the code using a Python sandbox
4. Collects the output and determines if the task is complete
5. Answer the original question based on the collected information

## ⚠️ Limitations

### Only accepts pure functions as tools (no callable objects)

The following example does not work due to the usage of a callable object.

```python
# ❌ NG
class Add():
    def __call__(self, a: int, b: int):
        return a + b

dspy.CodeAct("question -> answer", tools=[Add()])
```

### External libraries cannot be used

The following example does not work due to the usage of the external library `numpy`.

```python
# ❌ NG
import numpy as np

def exp(i: int):
    return np.exp(i)

dspy.CodeAct("question -> answer", tools=[exp])
```

### All dependent functions need to be passed to `CodeAct`

Functions that depend on other functions or classes not passed to `CodeAct` cannot be used. The following example does not work because the tool functions depend on other functions or classes that are not passed to `CodeAct`, such as `Profile` or `secret_function`.

```python
# ❌ NG
from pydantic import BaseModel

class Profile(BaseModel):
    name: str
    age: int
    
def age(profile: Profile):
    return 

def parent_function():
    print("Hi!")

def child_function():
    parent_function()

dspy.CodeAct("question -> answer", tools=[age, child_function])
```

Instead, the following example works since all necessary tool functions are passed to `CodeAct`:

```python
# ✅ OK

def parent_function():
    print("Hi!")

def child_function():
    parent_function()

dspy.CodeAct("question -> answer", tools=[parent_function, child_function])
```
