---
sidebar_position: 4
---

# dspy.cot

### Overview

#### `def cot(func) -> dspy.Module`

The `@cot` decorator is used to create a Chain of Thoughts module based on the provided function. It automatically generates a `dspy.TypedPredictor` and from the function's type annotations and docstring. Similar to predictor, but adds a "Reasoning" output field to capture the model's step-by-step thinking.

* **Input**: Function with input parameters and return type annotation.
* **Output**: A dspy.Module instance capable of making predictions.

### Example

```python
import dspy

context = ["Roses are red.", "Violets are blue"]
question = "What color are roses?"

@dspy.cot
def generate_answer(self, context: list[str], question) -> str:
    """Answer questions with short factoid answers."""
    pass

generate_answer(context=context, question=question)
```