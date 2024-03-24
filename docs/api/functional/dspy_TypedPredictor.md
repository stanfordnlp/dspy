---
sidebar_position: 1
---

# dspy.TypedPredictor

The `TypedPredictor` class is a sophisticated module designed for making predictions with strict type validations. It leverages a signature to enforce type constraints on inputs and outputs, ensuring that the data follows to the expected schema.

### Constructor

```python
TypedPredictor(
    CodeSignature
    max_retries=3
)
```

Parameters:
* `signature` (dspy.Signature): The signature that defines the input and output fields along with their types.
* `max_retries` (int, optional): The maximum number of retries for generating a valid prediction output. Defaults to 3.

### Methods

#### `copy() -> "TypedPredictor"`

Creates and returns a deep copy of the current TypedPredictor instance.

**Returns:** A new instance of TypedPredictor that is a deep copy of the original instance.

#### `_make_example(type_: Type) -> str`

A static method that generates a JSON object example based pn the schema of the specified Pydantic model type. This JSON object serves as an example for the expected input or output format.

**Parameters:**
* `type_`: A Pydantic model class for which an example JSON object is to be generated.

**Returns:** A string that represents a JSON object example, which validates against the provided Pydantic model's JSON schema. If the method is unable to generate a valid example, it returns an empty string.

#### `_prepare_signature() -> dspy.Signature`

Prepares and returns a modified version of the signature associated with the TypedPredictor instance. This method iterates over the signature's fields to add format and parser functions based on their type annotations.

**Returns:** A dspy.Signature object that has been enhanced with formatting and parsing specifications for its fields.

#### `forward(**kwargs) -> dspy.Prediction`

Executes the prediction logic, making use of the `dspy.Predict` component to generate predictions based on the input arguments. This method handles type validation, parsing of output data, and implements retry logic in case the output does not initially follow to the specified output schema.

**Parameters:**

* `**kwargs`: Keyword arguments corresponding to the input fields defined in the signature.

**Returns:** A dspy.Prediction object containing the prediction results. Each key in this object corresponds to an output field defined in the signature, and its value is the parsed result of the prediction.

### Example

```python
from dspy import InputField, OutputField, Signature
from dspy.functional import TypedPredictor
from pydantic import BaseModel

# We define a pydantic type that automatically checks if it's argument is valid python code.
class CodeOutput(BaseModel):
    code: str
    api_reference: str

class CodeSignature(Signature):
    function_description: str = InputField()
    solution: CodeOutput = OutputField()

cot_predictor = TypedPredictor(CodeSignature)
prediction = cot_predictor(
    function_description="Write a function that adds two numbers."
)

print(prediction["code"])
print(prediction["api_reference"])
```