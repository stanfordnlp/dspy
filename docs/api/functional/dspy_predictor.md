---
sidebar_position: 3
---

# dspy.predictor

### Overview

#### `def predictor(func) -> dspy.Module`

The `@predictor` decorator is used to create a predictor module based on the provided function. It automatically generates a `dspy.TypedPredictor` and from the function's type annotations and docstring.

* **Input**: Function with input parameters and return type annotation.
* **Output**: A dspy.Module instance capable of making predictions.

### Example

```python
import dspy

context = ["Roses are red.", "Violets are blue"]
question = "What color are roses?"

@dspy.predictor
def generate_answer(self, context: list[str], question) -> str:
    """Answer questions with short factoid answers."""
    pass

generate_answer(context=context, question=question)
```