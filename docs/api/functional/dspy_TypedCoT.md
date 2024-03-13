---
sidebar_position: 2
---

# dspy.TypedChainOfThought

### Overview

#### `def TypedChainOfThought(signature, max_retries=3) -> dspy.Module`

Adds a Chain of Thoughts `dspy.OutputField` to the `dspy.TypedPredictor` module by prepending it to the Signature. Similar to `dspy.TypedPredictor` but automatically adds a "reasoning" output field.

* **Inputs**:
    * `signature`: The `dspy.Signature` specifying the input/output fields
    * `max_retries`: Maximum number of retries if outputs fail validation
* **Output**: A dspy.Module instance capable of making predictions.

### Example

```python
from dspy import InputField, OutputField, Signature
from dspy.functional import TypedChainOfThought
from pydantic import BaseModel

# We define a pydantic type that automatically checks if it's argument is valid python code.
class CodeOutput(BaseModel):
    code: str
    api_reference: str

class CodeSignature(Signature):
    function_description: str = InputField()
    solution: CodeOutput = OutputField()

cot_predictor = TypedChainOfThought(CodeSignature)
prediction = cot_predictor(
    function_description="Write a function that adds two numbers."
)

print(prediction["code"])
print(prediction["api_reference"])
```