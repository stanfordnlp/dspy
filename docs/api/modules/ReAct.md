# dspy.ReAct

### Constructor

The constructor initializes the `ReAct` class and sets up its attributes. It is specifically designed to compose the interleaved steps of Thought, Action, and Observation.



```python
import dsp
import dspy
from ..primitives.program import Module
from .predict import Predict

class ReAct(Module):
    def __init__(self, signature, max_iters=5, num_results=3, tools=None):
        ...
```

**Parameters:**
- `signature` (_Any_): Signature of the predictive model.
- `max_iters` (_int_, _optional_): Maximum number of iterations for the Thought-Action-Observation cycle. Defaults to `5`.
- `num_results` (_int_, _optional_): Number of results to retrieve in the action step. Defaults to `3`.
- `tools` (_List[dspy.Tool]_, _optional_): List of tools available for actions. If none is provided, a default `Retrieve` tool with `num_results` is used.

### Methods

#### `_generate_signature(self, iters)`

Generates a signature for the Thought-Action-Observation cycle based on the number of iterations.

**Parameters:**
- `iters` (_int_): Number of iterations.

**Returns:**
- A dictionary representation of the signature.

***

#### `act(self, output, hop)`

Processes an action and returns the observation or final answer.

**Parameters:**
- `output` (_dict_): Current output from the Thought.
- `hop` (_int_): Current iteration number.

**Returns:**
- A string representing the final answer or `None`.

***

#### `forward(self, **kwargs)`

Main method to execute the Thought-Action-Observation cycle for a given set of input fields.

**Parameters:**
- `**kwargs`: Keyword arguments corresponding to input fields.

**Returns:**
- A `dspy.Prediction` object containing the result of the ReAct process.

### Examples

```python
# Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Pass signature to ReAct module
react_module = dspy.ReAct(BasicQA)

# Call the ReAct module on a particular input
question = 'What is the color of the sky?'
result = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {result.answer}")
```